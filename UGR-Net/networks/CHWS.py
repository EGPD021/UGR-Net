# LCAM
import torch
import torch.nn as nn
from torch.nn import init
import math
import numpy as np
class CHWS(nn.Module):
    def __init__(self, in_planes, ratio=8, gamma=2, b=1, pattern=3):
        super(CHWS, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes = in_planes
        kernel_size = int(abs((math.log(self.in_planes, 2) + b) / gamma))
        kernel_size = np.max([kernel_size, 3])
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,1,kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.act1 = nn.Sigmoid()
        self.pattern = pattern
    def forward(self, x):
        if self.pattern == 0:
           out1 = self.avg_pool(x) + self.max_pool(x)
        elif self.pattern == 1:
            out1 = self.avg_pool(x)
        elif self.pattern == 2:
            out1 = self.max_pool(x)
        else:
            output1 = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
            output1 = self.con1(output1).transpose(-1, -2).unsqueeze(-1)
        output2 = self.max_pool(x).squeeze(-1).transpose(-1, -2)
        output2 = self.con1(output2).transpose(-1, -2).unsqueeze(-1)
        out1 = output1 + output2
        if self.pattern != 3:
            out1 = out1.squeeze(-1).transpose(-1, -2)
            out1 = self.con1(out1).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(out1)
        return output