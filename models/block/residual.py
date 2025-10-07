import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)) + identity)

        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

        self.conv3 = nn.Conv2d(c_out, c_out * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)) + identity)

        return out
    
class ResdualBlock(nn.Module):
    def __init__(self, c_in, c_out, s):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, 
                               stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_in)
        
        self.conv2 = nn.Conv2d(c_in, c_in, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_in)
        
        self.conv3 = nn.Conv2d(c_in, c_out, kernel_size=1, 
                               stride=1, bias=False)
        
        self.proj = nn.Conv2d(c_out, c_out, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)) + self.proj(x))
        return x

