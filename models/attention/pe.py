import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
    def forward(self, x):
        x = self.proj(x)                       # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return x
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_h=256, max_w=256, learnable=False, temperature=10000, flat=True):
        super().__init__()
        if not learnable:
            assert dim % 4 == 0

        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.temperature = temperature
        self.flat = flat

        if learnable:
            self.pe = nn.Parameter(torch.randn(dim, max_h, max_w) * 0.02)
        else:
            self.register_buffer("pe", self._build_pe(), persistent=False)

    def _build_pe(self):
        dim_quarter = self.dim // 4

        y_pos = torch.arange(self.max_h, dtype=torch.float32).unsqueeze(1)  # (H,1)
        x_pos = torch.arange(self.max_w, dtype=torch.float32).unsqueeze(1)  # (W,1)

        div_term = torch.exp(
            torch.arange(dim_quarter, dtype=torch.float32) * -(math.log(self.temperature) / dim_quarter)
        )

        y_scaled = y_pos * div_term.unsqueeze(0)
        x_scaled = x_pos * div_term.unsqueeze(0)

        sin_y = torch.sin(y_scaled).unsqueeze(2).repeat(1, 1, self.max_w)
        cos_y = torch.cos(y_scaled).unsqueeze(2).repeat(1, 1, self.max_w)
        sin_x = torch.sin(x_scaled).unsqueeze(0).repeat(self.max_h, 1, 1)
        cos_x = torch.cos(x_scaled).unsqueeze(0).repeat(self.max_h, 1, 1)

        pe = torch.zeros(self.dim, self.max_h, self.max_w, dtype=torch.float32)
        pe[0:dim_quarter] = sin_y.permute(1, 0, 2)
        pe[dim_quarter:2*dim_quarter] = cos_y.permute(1, 0, 2)
        pe[2*dim_quarter:3*dim_quarter] = sin_x.permute(2, 0, 1)
        pe[3*dim_quarter:] = cos_x.permute(2, 0, 1)
        return pe

    def forward(self, B, H, W, device=None, dtype=None):
        assert H <= self.max_h and W <= self.max_w
        pe = self.pe[:, :H, :W]
        if not isinstance(self.pe, nn.Parameter):
            if device is not None or dtype is not None:
                pe = pe.to(device=device, dtype=dtype)

        if self.flat:
            pe = pe.view(self.dim, H * W).unsqueeze(0).expand(B, -1, -1)
            return pe
        return pe.unsqueeze(0).expand(B, -1, -1, -1)