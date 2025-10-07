import torch.nn as nn

from ..attention import SAM

class BottleneckTransformer(nn.Module):
    def __init__(self, c_in, c_out, heads):
        super().__init__()

        self.mhsa1 = SAM(c_in, c_out, heads)
        self.bn1 = nn.BatchNorm2d(c_out)

        self.mhsa2 = SAM(c_out, c_out, heads)
        self.bn2 = nn.BatchNorm2d(c_out)

        self.proj = nn.Conv2d(c_out, c_out, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.mhsa1(x)))
        x = self.relu(self.bn2(self.mhsa2(x)) + self.proj(x))

        return x

