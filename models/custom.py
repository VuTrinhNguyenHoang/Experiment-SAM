import torch
import torch.nn as nn

from .block import BasicBlock, BottleneckBlock, ResdualBlock, BottleneckTransformer, SAMBlock
from .lstm import xLSTM
from .backbone import get_pretrained_model
from .attention import SAMv2

class miniARCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.block1 = ResdualBlock(8, 18, 1)
        self.block2 = ResdualBlock(18, 28, 2)
        self.block3 = BottleneckTransformer(28, 8, 4)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8*7*7, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class miniARCNN_xLSTM(nn.Module):
    def __init__(self, num_classes, layers=['s','m'], dropout=0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.block1 = ResdualBlock(8, 18, 1)
        self.block2 = ResdualBlock(18, 28, 2)
        self.block3 = BottleneckTransformer(28, 8, 4)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.xlstm = xLSTM(input_dim=8, hidden_dim=16, layers=layers)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = self.xlstm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)
    
class ViTSAM(nn.Module):
    def __init__(
        self,
        model_name="vit_small_patch16_224",
        num_classes=7,
        pretrained=True,
        in_chans=3,
        sam_type="all"
    ):
        super().__init__()
        self.model = get_pretrained_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans
        )

        total_blocks = len(self.model.blocks)
        if sam_type == "all":
            sam_blocks = list(range(0, total_blocks))
        elif sam_type == "hybrid":
            sam_blocks = list(range(1, total_blocks, 2))

        for i in sam_blocks:
            blk = self.model.blocks[i]
            embed_dim = blk.attn.qkv.in_features
            num_heads = blk.attn.num_heads
            blk.attn = SAMv2(embed_dim, num_heads, drop=0.1, attn_drop=0.1, bidirectional=True)

    def forward(self, x):
        return self.model(x)

class ViTSAMxLSTM(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", num_classes=7,
                 pretrained=True, in_chans=3, sam_type="hybrid",
                 bottleneck_dim=256, xlstm_hidden=256, xlstm_layers=('s','m'),
                 dropout=0.1):
        super().__init__()
        self.backbone = get_pretrained_model(model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans)
        self.embed_dim = getattr(self.backbone, "embed_dim")

        total_blocks = len(self.backbone.blocks)
        sam_blocks = list(range(0, total_blocks)) if sam_type=="all" else list(range(1, total_blocks, 2))
        for i in sam_blocks:
            blk = self.backbone.blocks[i]
            D = blk.attn.qkv.in_features
            Hs = blk.attn.num_heads
            blk.attn = SAMv2(embed_dim=D, num_heads=Hs, drop=0.1, attn_drop=0.1, bidirectional=True)

        # ↓ chiếu D → d trước xLSTM
        self.proj_in = nn.Linear(self.embed_dim, bottleneck_dim)

        # xLSTM ở không gian d
        self.xlstm = xLSTM(input_dim=bottleneck_dim, hidden_dim=xlstm_hidden, layers=xlstm_layers)

        # head làm trên d (đặt hidden=bottleneck để bỏ proj_out)
        self.proj_out = nn.Identity() if xlstm_hidden == bottleneck_dim else nn.Linear(xlstm_hidden, bottleneck_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(bottleneck_dim, num_classes)

    def forward_features_tokens(self, x):
        b = x.size(0)
        x = self.backbone.patch_embed(x)                                  # B,N,D
        if getattr(self.backbone, "cls_token", None) is not None:
            cls_tok = self.backbone.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tok, x), dim=1)                            # B,1+N,D
        if getattr(self.backbone, "pos_embed", None) is not None:
            x = x + self.backbone.pos_embed[:, :x.size(1), :]
        if getattr(self.backbone, "pos_drop", None) is not None:
            x = self.backbone.pos_drop(x)
        for blk in self.backbone.blocks:
            x = blk(x)                                                    # B,1+N,D
        x = self.backbone.norm(x)                                         # B,1+N,D
        return x

    def forward(self, x):
        tok = self.forward_features_tokens(x)                              # B,N,D
        tok = self.proj_in(tok)                                            # B,N,d
        tok = self.xlstm(tok)                                              # B,N,H
        tok = self.proj_out(tok)                                           # B,N,d
        feat = tok[:, 0, :]                                                # CLS sau xLSTM
        return self.head(self.dropout(feat))

class ResNetSAM(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, in_chans=3, num_heads=4, dropout=0.0):
        super().__init__()
        self.model = get_pretrained_model(
            model_name, 
            num_classes=0, 
            pretrained=pretrained, 
            in_chans=in_chans
        )
        
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Identity()

        last_stage = self.model.layer4
        last_block = last_stage[-1]
        c_out = self._last_block_out_channels(last_block)
        heads = num_heads

        new_blocks = list(last_stage.children()) + [SAMBlock(c_out, heads=heads)]
        self.model.layer4 = nn.Sequential(*new_blocks)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(c_out, num_classes)

    def _last_block_out_channels(self, block):
        for attr in ["conv3", "conv2"]:
            if hasattr(block, attr):
                return getattr(block, attr).out_channels

        if hasattr(block, "out_channels"):
            return getattr(block, "out_channels")
        
        raise ValueError("Không xác định được số kênh out của block cuối.")
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.head(x)
        return x

class ResNetSAMxLSTM(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, in_chans=3,
                 num_heads=4, bottleneck_dim=256, xlstm_hidden=256, xlstm_layers=('s','m'),
                 dropout=0.0):
        super().__init__()
        self.model = get_pretrained_model(model_name, num_classes=0,
                                          pretrained=pretrained, in_chans=in_chans)
        if hasattr(self.model, "fc"):
            self.model.fc = nn.Identity()

        last_stage = self.model.layer4
        last_block = last_stage[-1]
        c_out = self._last_block_out_channels(last_block)

        self.model.layer4 = nn.Sequential(*list(last_stage.children()),
                                          SAMBlock(c_out, heads=num_heads))

        # ↓ chiếu C_out → d (nhỏ) trước khi vào xLSTM
        self.proj_in = nn.Linear(c_out, bottleneck_dim)

        # xLSTM chạy ở không gian d (nhỏ)
        self.seq = xLSTM(input_dim=bottleneck_dim, hidden_dim=xlstm_hidden, layers=xlstm_layers)

        # head làm trên hidden (chọn bằng với bottleneck để gọn)
        self.proj_out = nn.Identity() if xlstm_hidden == bottleneck_dim else nn.Linear(xlstm_hidden, bottleneck_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(bottleneck_dim, num_classes)

    def _last_block_out_channels(self, block):
        for attr in ["conv3", "conv2"]:
            if hasattr(block, attr):
                return getattr(block, attr).out_channels

        if hasattr(block, "out_channels"):
            return getattr(block, "out_channels")
        
        raise ValueError("Không xác định được số kênh out của block cuối.")

    def forward(self, x):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.act1(x); x = self.model.maxpool(x)
        x = self.model.layer1(x); x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)  # B,C,H,W

        tok = x.flatten(2).transpose(1, 2)     # B,T,C_out
        tok = self.proj_in(tok)                # B,T,d
        tok = self.seq(tok)                    # B,T,H
        tok = self.proj_out(tok)               # B,T,d
        feat = tok.mean(dim=1)                 # B,d
        return self.head(self.dropout(feat))
    
class VGG19SAM(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'vgg19_bn',
                 pretrained: bool = True, in_chans: int = 3,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # Use timm backbone without classifier head
        self.model = get_pretrained_model(
            model_name,
            num_classes=0,
            pretrained=pretrained,
            in_chans=in_chans
        )

        # Try to remove existing classifier if present
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()

        c_out = getattr(self.model, 'num_features')
        self.sam = SAMBlock(c_out, heads=num_heads)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(c_out, num_classes)

    def forward(self, x):
        # Generic CNN path in timm: forward_features -> 4D map
        x = self.model.forward_features(x)   # B,C,H,W
        x = self.sam(x)                      # B,C,H,W
        x = self.model.global_pool(x)        # B,C,1,1 or B,C
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)

class VGG19SAMxLSTM(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'vgg19_bn',
                 pretrained: bool = True, in_chans: int = 3,
                 num_heads: int = 4,
                 bottleneck_dim: int = 256, xlstm_hidden: int = 256,
                 xlstm_layers=('s', 'm'), dropout: float = 0.0):
        super().__init__()
        self.model = get_pretrained_model(
            model_name,
            num_classes=0,
            pretrained=pretrained,
            in_chans=in_chans
        )

        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()

        c_out = getattr(self.model, 'num_features')
        self.sam = SAMBlock(c_out, heads=num_heads)

        # Project channel features to a compact token dim for sequence modeling
        self.proj_in = nn.Linear(c_out, bottleneck_dim)
        self.seq = xLSTM(input_dim=bottleneck_dim, hidden_dim=xlstm_hidden, layers=xlstm_layers)
        self.proj_out = nn.Identity() if xlstm_hidden == bottleneck_dim else nn.Linear(xlstm_hidden, bottleneck_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x):
        # CNN features -> SAM -> tokens -> xLSTM -> mean pool tokens
        f = self.model.forward_features(x)   # B,C,H,W
        f = self.sam(f)                      # B,C,H,W
        tok = f.flatten(2).transpose(1, 2)   # B,T,C
        tok = self.proj_in(tok)              # B,T,d
        tok = self.seq(tok)                  # B,T,H
        tok = self.proj_out(tok)             # B,T,d
        feat = tok.mean(dim=1)               # B,d
        return self.head(self.dropout(feat))

class MobileViTSAM(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'mobilevit_s',
                 pretrained: bool = True, in_chans: int = 3,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.model = get_pretrained_model(
            model_name,
            num_classes=0,
            pretrained=pretrained,
            in_chans=in_chans
        )

        c_out = getattr(self.model, 'num_features')
        self.sam = SAMBlock(c_out, heads=num_heads)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(c_out, num_classes)

        # If the base model exposes a classifier, neutralize it to avoid accidental use
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()

    def forward(self, x):
        f = self.model.forward_features(x)   # B,C,H,W for MobileViT
        f = self.sam(f)
        f = self.model.global_pool(f)
        f = torch.flatten(f, 1)
        f = self.dropout(f)
        return self.head(f)

class MobileViTSAMxLSTM(nn.Module):
    def __init__(self, num_classes: int, model_name: str = 'mobilevit_s',
                 pretrained: bool = True, in_chans: int = 3,
                 num_heads: int = 4,
                 bottleneck_dim: int = 256, xlstm_hidden: int = 256,
                 xlstm_layers=('s','m'), dropout: float = 0.0):
        super().__init__()
        self.model = get_pretrained_model(
            model_name,
            num_classes=0,
            pretrained=pretrained,
            in_chans=in_chans
        )

        c_out = getattr(self.model, 'num_features')
        self.sam = SAMBlock(c_out, heads=num_heads)

        self.proj_in = nn.Linear(c_out, bottleneck_dim)
        self.seq = xLSTM(input_dim=bottleneck_dim, hidden_dim=xlstm_hidden, layers=xlstm_layers)
        self.proj_out = nn.Identity() if xlstm_hidden == bottleneck_dim else nn.Linear(xlstm_hidden, bottleneck_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(bottleneck_dim, num_classes)

        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()

    def forward(self, x):
        f = self.model.forward_features(x)   # B,C,H,W
        f = self.sam(f)
        tok = f.flatten(2).transpose(1, 2)   # B,T,C
        tok = self.proj_in(tok)              # B,T,d
        tok = self.seq(tok)                  # B,T,H
        tok = self.proj_out(tok)             # B,T,d
        feat = tok.mean(dim=1)               # B,d
        return self.head(self.dropout(feat))
    
