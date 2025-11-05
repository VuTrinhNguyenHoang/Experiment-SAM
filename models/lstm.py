import torch
import torch.nn as nn

class sLSTMblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(input_dim, 4 * hidden_dim)
        self.U = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.alpha = nn.Parameter(torch.ones(hidden_dim))
        self.do = nn.Dropout(dropout)

    def forward(self, x, hc=None):
        B, T, D = x.size()
        H = self.U.out_features // 4
        h, c = (x.new_zeros(B, H), x.new_zeros(B, H)) if hc is None else hc
        outs = []
        for t in range(T):
            i, f, o, g = (self.W(x[:, t]) + self.U(h)).chunk(4, -1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = self.alpha * (f * c + i * g)
            h = o * torch.tanh(c)
            outs.append(self.do(h).unsqueeze(1))
        return torch.cat(outs, 1), (h, c)
    
class mLSTMblock(nn.Module):
    def __init__(self, input_dim, hidden_dim, rank=64, dropout=0.1):
        super().__init__()
        self.W = nn.Linear(input_dim, 4 * hidden_dim)
        self.U = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(hidden_dim, rank, bias=False)
        self.P = nn.Linear(rank, hidden_dim, bias=False)
        self.do = nn.Dropout(dropout)

    def forward(self, x, hc=None):
        B, T, _ = x.size(); H = self.U.out_features // 4
        h = x.new_zeros(B, H) if hc is None else hc[0]
        c = x.new_zeros(B, H) if hc is None else hc[1]
        outs = []
        for t in range(T):
            i, f, o, g = (self.W(x[:, t]) + self.U(h)).chunk(4, -1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            mix = self.P(self.A(x[:, t]) * self.B(h))   # O(rank*(in+H))
            c = f * c + i * g + 0.1 * mix
            h = o * torch.tanh(c)
            outs.append(self.do(h).unsqueeze(1))
        return torch.cat(outs, 1), (h, c)
    
class xLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=['s','m'], 
                 rank=64, dropout=0.1, proj_out=None):
        super().__init__()
        blocks = []
        d = input_dim
        for l in layers:
            blk = sLSTMblock(d, hidden_dim, dropout) if l == 's' else mLSTMblock(d, hidden_dim, rank, dropout)
            blocks.append(blk); d = hidden_dim
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Linear(d, proj_out) if proj_out is not None else None

    def forward(self, x):
        out = x
        for blk in self.blocks:
            out, _ = blk(out)
        return self.proj(out) if self.proj is not None else out