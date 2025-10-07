import torch
import timm

def get_pretrained_model(model_name, num_classes, pretrained=False):
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )

def inspect_model(model, inputs=(1, 3, 224, 224)):
    x = torch.randn(inputs)
    print(f"{'Layer':25s} {'Output Shape':20s} {'Params':10s}")
    print("-" * 60)

    for name, layer in model.named_children():
        x = layer(x)
        params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"{name:25s} {str(list(x.shape)):20s} {params/1e6:8.3f}M")

    print("-" * 60)
    print("Final feature map:", list(x.shape))

__all__ = [
    'get_pretrained_model',
    'inspect_model'
]