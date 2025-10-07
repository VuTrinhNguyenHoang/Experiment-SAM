import torch
import timm
from collections import OrderedDict

def get_pretrained_model(model_name, num_classes, pretrained=False):
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

def _num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def inspect_model(model, inputs=(1, 3, 224, 224), print_summary=True, return_details=False):
    model.eval()
    dummy = torch.randn(inputs)

    call_records = []
    name_map = {}
    for name, m in model.named_modules():
        name_map[m] = name

    def hook(module, inp, out):
        if module is model:
            return
        name = name_map.get(module, module.__class__.__name__)
        shape = list(out.shape) if isinstance(out, torch.Tensor) else [type(out)]
        call_records.append({
            'name': name,
            'class': module.__class__.__name__,
            'out_shape': shape,
            'params': _num_params(module)
        })

    hooks = []
    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook))

    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    if print_summary:
        print(f"{'Idx':>3s} {'Layer':35s} {'Class':20s} {'Output Shape':20s} {'Params(M)':>10s}")
        print('-'*100)
        for idx, rec in enumerate(call_records):
            print(f"{idx:3d} {rec['name'][:35]:35s} {rec['class'][:20]:20s} {str(rec['out_shape'])[:20]:20s} {rec['params']/1e6:10.4f}")
        print('-'*100)
        total_params = _num_params(model)
        print(f"Total trainable params: {total_params/1e6:.3f} M")
        if call_records:
            print("Final feature map shape:", call_records[-1]['out_shape'])

    return call_records if return_details else None

__all__ = [
    'get_pretrained_model',
    'inspect_model'
]