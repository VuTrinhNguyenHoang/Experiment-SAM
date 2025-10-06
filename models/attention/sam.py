import torch
import torch.nn as nn
import math

from .pe import PositionalEncoding2D

class SAM(nn.Module):
    def __init__(self, c_in, c_out, heads=4):
        super().__init__()
        assert c_out % heads == 0
        self.heads = heads
        self.dh = c_out // heads

        self.v_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.z_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.q_proj = nn.Conv2d(c_in, c_out, 1, bias=False)

        self.pe = PositionalEncoding2D(c_out, learnable=False)

    def forward(self, x):
        B, _, H, W = x.shape
        M = H * W
        v = self.v_proj(x).flatten(2)
        z = self.z_proj(x).flatten(2)
        q = self.q_proj(x).flatten(2)
        r = self.pe(B, H, W)

        def split(t): 
            return t.view(B, self.heads, self.dh, M)
        vh, zh, qh, rh = map(split, (v, z, q, r))

        A = torch.softmax((torch.einsum('bhcm,bhcn->bhmn', vh, zh)
                           + torch.einsum('bhcm,bhcn->bhmn', vh, rh)) / math.sqrt(self.dh), dim=-1)
        out = torch.einsum('bhmn,bhdn->bhdm', A, qh)
        return out.reshape(B, -1, M).view(B, -1, H, W)