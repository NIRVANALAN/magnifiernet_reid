import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from ..Tools import *

### Segmentation Related
def SegPred(seg_res, num_classes=8):
    #print(seg_res.size()) #bs C H W
    return torch.max(seg_res, dim=1)[1]

class PartSegModule(nn.Module):
    def __init__(self, in_channels, mid_c, num_classes):
        super(PartSegModule, self).__init__()

        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=mid_c,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Conv2d(
            in_channels=mid_c,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        nn.init.normal_(self.deconv.weight, std=0.001)
        nn.init.normal_(self.conv.weight, std=0.001)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(self.deconv(x))))
        return x
    
