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
    
class SAMv2(nn.Module):
    def __init__(self, embed_dim, num_heads, drop: float = 0.0, attn_drop: float = 0.0, bidirectional=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.sam = SAM(c_in=embed_dim, c_out=embed_dim, heads=num_heads)

        self.cls_from_patch = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.patch_from_cls = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        cls = x[:, :1, :]
        tok = x[:, 1:, :]
        HW = tok.size(1)
        H = W = int(math.sqrt(HW))

        tok_2d = tok.transpose(1, 2).reshape(B, D, H, W)
        y_2d = self.sam(tok_2d)

        y = y_2d.flatten(2).transpose(1, 2)
        g = y.mean(dim=1, keepdim=True)
        cls = cls + self.cls_from_patch(g)
        
        if self.bidirectional:
            y = y + self.patch_from_cls(cls).expand_as(y)

        out = torch.cat([cls, y], dim=1)
        out = self.attn_drop(out)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SAM_R(nn.Module):
    def __init__(self, c_in, c_out, num_heads, attn_drop, proj_drop, learnable=False):
        super().__init__()
        assert c_out % num_heads == 0
        self.h = num_heads
        self.dh = c_out // self.h
        
        self.norm = nn.BatchNorm2d(c_in)
        self.v = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.z = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.q = nn.Conv2d(c_in, c_out, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(c_out, c_out, 1, bias=False)
        self.out_drop = nn.Dropout2d(proj_drop)
        nn.init.zeros_(self.out_proj.weight)
        
        self.pe = PositionalEncoding2D(c_out, learnable=learnable, flat=False)

    def _split(self, t):
        B, C, H, W = t.shape
        M = H * W
        t = t.view(B, self.h, self.dh, M)
        return t, H, W
    
    def l2_norm_per_channel_token(self, x, eps: float = 1e-6):
        n = x.pow(2).sum(1, keepdim=True).add_(eps).sqrt_()
        return x / n

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.norm(x)
        v, z, q = self.v(x), self.z(x), self.q(x)
        r = self.pe(B, H, W, x.device, x.dtype)

        zN = self.l2_norm_per_channel_token(z)
        rN = self.l2_norm_per_channel_token(r)

        zr = zN + rN
        vh, _, _ = self._split(v)
        zrh, _, _ = self._split(zr)
        qh, _, _ = self._split(q)

        logits = torch.einsum('bhcm,bhcn->bhmn', vh, zrh) / math.sqrt(self.dh)
        A = torch.softmax(logits, dim=-1)
        A = self.attn_drop(A)

        y = torch.einsum('bhmn,bhcn->bhcm', A, qh)  # (B,h,dh,M)
        y = y.view(B, self.h * self.dh, H, W)
        y = self.out_proj(y)
        y = self.out_drop(y)
        return y
