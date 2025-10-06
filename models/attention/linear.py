import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, c_in, c_out, heads=4):
        super().__init__()
        self.heads = heads
        self.dh = c_out // heads

        self.q_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.k_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.v_proj = nn.Conv2d(c_in, c_out, 1, bias=False)

    def feature_map(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, _, H, W = x.shape
        M = H * W

        q = self.feature_map(self.q_proj(x).flatten(2).view(B, self.heads, self.dh, M))
        k = self.feature_map(self.k_proj(x).flatten(2).view(B, self.heads, self.dh, M))
        v = self.v_proj(x).flatten(2).view(B, self.heads, self.dh, M)

        kv = torch.einsum("bhdk,bhdm->bhkm", k, v)
        z = 1 / (torch.einsum("bhdk,bhdm->bhm", q, k.sum(dim=-1, keepdim=True)) + 1e-6)
        out = torch.einsum("bhdk,bhkm->bhdm", q, kv) * z.unsqueeze(1)
        return out.reshape(B, -1, H, W)
