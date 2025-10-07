import torch
import torch.nn as nn

from .block import BasicBlock, BottleneckBlock, ResdualBlock, BottleneckTransformer
from .lstm import xLSTM

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
    def __init__(self, num_classes, layers=['s','m']):
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
        self.xlstm = xLSTM(input_dim=8, hidden_dim=8*2, layers=layers)

        self.fc = nn.Linear(8*2, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(0, 2, 1)

        x = self.xlstm(x)
        x = x.mean(dim=1)
        
        return self.fc(x)
    
class AttentionAdapter(nn.Module):
    def __init__(self, backbone, attn_cfg):
        """
        attn_cfg: list[tuple(str, nn.Module)]
            [(layer_name, attention_module), ...]
        """
    
        super().__init__()
        self.backbone = backbone
        self.attn_cfg = {k: v for k, v in attn_cfg}

    def forward(self, x):
        for name, layer in self.backbone.named_children():
            x = layer(x)
            if name in self.attn_cfg:
                x = self.attn_cfg[name](x)
        return x
    
