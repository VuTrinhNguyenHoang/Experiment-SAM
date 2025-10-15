import torch
import torch.nn as nn

from .block import BasicBlock, BottleneckBlock, ResdualBlock, BottleneckTransformer
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
            blk.attn = SAMv2(embed_dim, num_heads)

    def freeze_backbone_except_sam(self):
        for p in self.model.parameters():
            p.requires_grad = False
        
        for name, m in self.model.named_modules():
            if "attn" in name or "sam" in name:
                for p in m.parameters():
                    p.requires_grad = True

        for p in self.model.head.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
