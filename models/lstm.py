import torch
import torch.nn as nn

class sLSTMblock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, 4 * hidden_dim)
        self.U = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.alpha = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, hc=None):
        B, T, _ = x.size()
        if hc is None:
            h = torch.zeros(B, self.U.out_features // 4, device=x.device)
            c = torch.zeros(B, self.U.out_features // 4, device=x.device)
        else:
            h, c = hc

        outs = []
        for t in range(T):
            gates = self.W(x[:, t]) + self.U(h)
            i, f, o, g = gates.chunk(4, dim=-1)
            i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
            c = f * c + i * g
            c = self.alpha * c
            h = o * torch.tanh(c)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1), (h, c)
    
class mLSTMblock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W = nn.Linear(input_dim, 4 * hidden_dim)
        self.U = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.V = nn.Linear(input_dim * hidden_dim, hidden_dim)

    def forward(self, x, hc=None):
        B, T, _ = x.size()
        if hc is None:
            h = torch.zeros(B, self.U.out_features // 4, device=x.device)
            c = torch.zeros(B, self.U.out_features // 4, device=x.device)
        else:
            h, c = hc

        outs = []
        for t in range(T):
            gates = self.W(x[:, t]) + self.U(h)
            i, f, o, g = gates.chunk(4, dim=-1)
            i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)

            mult = (x[:, t].unsqueeze(2) * h.unsqueeze(1))        # (B, input_dim, hidden_dim)
            m = torch.tanh(self.V(mult.reshape(B, -1)))

            c = f * c + i * g + 0.1 * m
            h = o * torch.tanh(c)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1), (h, c)
    
class xLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=['s','m']):
        super().__init__()
        self.blocks = nn.ModuleList()
        for l in layers:
            if l == 's':
                self.blocks.append(sLSTMblock(input_dim, hidden_dim))
            elif l == 'm':
                self.blocks.append(mLSTMblock(input_dim, hidden_dim))
            else:
                raise ValueError("layer must be 's' or 'm'")
            input_dim = hidden_dim  # block sau nhận hidden_dim làm input

    def forward(self, x):
        out = x
        for block in self.blocks:
            out, _ = block(out)
        return out
    
