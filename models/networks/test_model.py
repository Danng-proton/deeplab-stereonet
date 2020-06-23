import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from correlation_package.modules.correlation import Correlation

from submodules import *
'Parameter count , 39,175,298 '

class FlowNetC(nn.Module):
    def __init__(self):
        super(FlowNetC,self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.corr = Correlation(pad_size=1, kernel_size=1, max_displacement=2, stride1=1, stride2=2, corr_multiply=1)

    def forward(self, x1, x2):
        x1=self.conv1(x1)
        x2=self.conv1(x2)
        out_corr = self.corr(x1, x2) 
        return out_corr
