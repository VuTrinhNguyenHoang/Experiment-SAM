import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, heads=4):
        super().__init__()
        self.heads = heads
        self.dh = d_model // heads
        self.scale = math.sqrt(self.dh)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, L, _ = q.size()
        q = self.q_proj(q).view(B, L, self.heads, self.dh).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.heads, self.dh).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.heads, self.dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out)
