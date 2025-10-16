import os
import torch
import numpy as np
from tqdm.auto import tqdm
from .benchmark import evaluate

def train_one_epoch(model, loader, criterion, optimizer, gpu_aug=None, MEAN=None, STD=None):
    device = next(model.parameters()).device
    model.train()

    total, correct, running = 0, 0, 0.0
    pbar = tqdm(loader, leave=False)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if gpu_aug != None:
            x = gpu_aug(x)
        if (MEAN != None) and (STD != None):
            x = (x - MEAN) / STD

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_postfix(loss=running/total, acc=correct/total)

    return running/total, correct/total

def save_ckpt(model, path, meta):
    if os.path.exists(path):
        os.remove(path)
    torch.save({"model": model.state_dict(), "meta": meta}, path)

def freeze_blocks(model, n_unfreeze=4):
    for p in model.parameters():
        p.requires_grad = False

    core = getattr(model, "model", model)

    def _open_params(m):
        if m is None: return
        for p in m.parameters(): p.requires_grad = True

    # 2) Mở classifier/head ở cả wrapper và core (head, fc, classifier)
    for hn in ("head", "fc", "classifier"):
        _open_params(getattr(model, hn, None))
        _open_params(getattr(core, hn, None))

    # 3) Luôn mở các mô-đun đặc biệt ở bất cứ đâu (SAM/xLSTM)
    SPECIALS = {"SAM", "SAMBlock", "xLSTM", "sLSTMblock", "mLSTMblock"}
    for m in core.modules():
        if m.__class__.__name__ in SPECIALS:
            _open_params(m)

    # 4) Nhánh theo kiến trúc

    # ViT/DeiT: có .blocks (timm/torchvision vit)
    if hasattr(core, "blocks"):
        try:
            blocks = list(core.blocks)
        except TypeError:
            # một số timm dùng Sequential
            blocks = list(core.blocks.children())
        for blk in blocks[-n_unfreeze:]:
            _open_params(blk)
        return

    # ResNet-family: có layer4
    if hasattr(core, "layer4") and isinstance(core.layer4, nn.Sequential):
        for blk in list(core.layer4)[-n_unfreeze:]:
            _open_params(blk)
        # Nếu wrapper có chuỗi sau backbone: seq/xlstm
        for an in ("seq", "xlstm"):
            _open_params(getattr(model, an, None))
        return

    # miniARCNN_xLSTM: mở block cuối + xlstm nếu có tên này
    for an in ("block3", "xlstm"):
        _open_params(getattr(model, an, None))

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def train_model(model_name, model, train_loader, valid_loader, criterion, optimizer, scheduler,
                gpu_aug=None, MEAN=None, STD=None, epochs=5, patience=None, eps=1e-4, warmup_epochs=5):
    best_loss, best_epoch = float('inf'), -1
    best_f1, best_acc = 0.0, 0.0
    best_path = f"{model_name}_best.pt"

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": []
    }
    no_improve_epochs = 0

    print(f"[INFO] Warmup finetuning last 4 blocks for {warmup_epochs} epochs...")
    freeze_blocks(model, n_unfreeze=4)

    pbar = tqdm(range(1, epochs+1), desc=model_name, unit="epoch")
    for epoch in pbar:
        if epoch == warmup_epochs + 1:
            print(f"[INFO] Unfreezing all layers from epoch {epoch}...")
            unfreeze_all(model)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, gpu_aug, MEAN, STD)
        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion, MEAN, STD)

        history["train_acc"].append(train_acc)
        history["valid_acc"].append(valid_acc)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        scheduler.step()

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", valid_loss=f"{valid_loss:.4f}",
                             valid_acc=f"{valid_acc*100:.2f}%", valid_f1=f"{valid_f1:.4f}")

        print(f"[{epoch}/{epochs}]: train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | valid_acc={valid_acc*100:.2f}% | valid_f1={valid_f1:.4f}")
        
        if patience is not None:
            # improved = best_loss == float('inf') or (best_loss - valid_loss) > eps
            improved = (best_epoch == -1) or ((valid_f1 - best_f1) > eps)
            if improved:
                best_loss, best_epoch, best_f1, best_acc = valid_loss, epoch, valid_f1, valid_acc
                save_ckpt(model, best_path, {"model_name": model_name, "epoch": epoch,
                                            "best_loss": best_loss, "best_f1": best_f1, "best_acc": best_acc})
                no_improve_epochs = 0
            else:
                if epoch <= warmup_epochs:
                    continue
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"EarlyStopping tại epoch {epoch}. Best: loss={best_loss:.4f} f1={best_f1:.4f} acc={best_acc*100:.2f}% ở epoch {best_epoch}")
                    break

    device = next(model.parameters()).device
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    _, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion, MEAN, STD)
    num_params = sum(p.numel() for p in model.parameters())

    return history, {
        "model_name": model_name,
        "valid_acc": valid_acc,
        "valid_f1": valid_f1,
        "num_params": num_params,
        "ckpt_path": best_path
    }

def class_weights(dataset):
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes))
    w = 1.0 / torch.tensor(counts, dtype=torch.float)
    w = w / w.sum() * len(counts)
    return w, counts

