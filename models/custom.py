import torch
import torch.nn as nn

from .block import BasicBlock, BottleneckBlock, ResdualBlock, BottleneckTransformer, SAMBlock
from .lstm import xLSTM
from .backbone import get_pretrained_model

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
    def __init__(self, num_classes, layers=['s','m'], bilinear_scale=0.1, dropout=0.0):
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
        self.xlstm = xLSTM(input_dim=8, hidden_dim=16, layers=layers, bilinear_scale=bilinear_scale)

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
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, in_chans=3, num_heads=4, xlstm_hidden=512, xlstm_layers=('s','m'),
                 bilinear_scale=0.1, dropout=0.0):
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

        self.seq = xLSTM(input_dim=c_out, hidden_dim=xlstm_hidden,
                         layers=xlstm_layers, bilinear_scale=bilinear_scale)

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

        x = x.flatten(2).transpose(1, 2)
        x = self.seq(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.head(x)
        return x