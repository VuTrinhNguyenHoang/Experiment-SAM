import torch
import torch.nn as nn

class CoordAttention(nn.Module):
    def __init__(self, c_in, reduction=32):
        super().__init__()
        c_mid = max(8, c_in // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c_in, c_mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(c_mid, c_in, 1, bias=False)
        self.conv_w = nn.Conv2d(c_mid, c_in, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        x_h = self.pool_h(x).permute(0, 1, 3, 2)  # (B, C, H, 1) -> (B, C, 1, H)
        x_w = self.pool_w(x)                      # (B, C, 1, W)

        y = torch.cat([x_h, x_w], dim=3)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [H, W], dim=3)
        x_h = self.conv_h(x_h.permute(0, 1, 3, 2))
        x_w = self.conv_w(x_w)

        out = x * torch.sigmoid(x_h) * torch.sigmoid(x_w)
        return out
