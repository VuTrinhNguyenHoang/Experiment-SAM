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
    def __init__(self, input_dim, hidden_dim, bilinear_bias=True, bilinear_scale=0.1):
        super().__init__()
        self.W = nn.Linear(input_dim, 4 * hidden_dim)
        self.U = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.B = nn.Bilinear(input_dim, hidden_dim, hidden_dim, bias=bilinear_bias)
        self.scale = bilinear_scale

        nn.init.xavier_uniform_(self.W.weight); nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.U.weight); nn.init.zeros_(self.U.bias)
        nn.init.xavier_uniform_(self.B.weight)
        if bilinear_bias: 
            nn.init.zeros_(self.B.bias)

    def forward(self, x, hc=None):
        B, T, D = x.size()
        H = self.U.out_features // 4
        if hc is None:
            h = x.new_zeros(B, H)
            c = x.new_zeros(B, H)
        else:
            h, c = hc

        outs = []
        for t in range(T):
            gates = self.W(x[:, t]) + self.U(h)
            i, f, o, g = gates.chunk(4, dim=-1)
            i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)

            # Bilinear interaction term m_t
            m = torch.tanh(self.B(x[:, t], h))          # (B, H)

            c = f * c + i * g + self.scale * m
            h = o * torch.tanh(c)
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1), (h, c)
    
class xLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=['s','m'], bilinear_scale=0.1):
        super().__init__()
        self.blocks = nn.ModuleList()
        for l in layers:
            if l == 's':
                self.blocks.append(sLSTMblock(input_dim, hidden_dim))
            elif l == 'm':
                self.blocks.append(mLSTMblock(input_dim, hidden_dim, bilinear_bias=True, bilinear_scale=bilinear_scale))
            else:
                raise ValueError("layer must be 's' or 'm'")
            input_dim = hidden_dim  # block sau nhận hidden_dim làm input

    def forward(self, x):
        out = x
        for block in self.blocks:
            out, _ = block(out)
        return out
    
