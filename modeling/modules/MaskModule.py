import torch.nn as nn
import math
import random
import torchvision
import torch
from torch.nn import functional as F
from torch.nn import init
#import linklink as link

from ..backbones.resnet import *
from ..Tools import weights_init_kaiming

class DoubleBatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(DoubleBatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)

            sx1 = random.randint(0, round((h - rh) / 2))
            sy1 = random.randint(0, w - rw)

            sx2 = random.randint(round((h - rh) / 2), h - rh)
            sy2 = random.randint(0, w - rw)
            mask1 = x.new_ones(x.size())
            mask1[:, :, sx1:sx1 + rh, sy1:sy1 + rw] = 0
            mask2 = x.new_ones(x.size())
            mask2[:, :, sx2:sx2 + rh, sy2:sy2 + rw] = 0
            x = x * mask1 * mask2
        return x


class SegmentBatchDrop(nn.Module):

    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        #rank = link.get_rank()
        if not isinstance(strategy, bool):
            if 'torso' in strategy:
                self.upper_torso = [1, 2, 3, 4]
                self.lower_torso = [5, 6, 7]
                self.erase_upper_number = int(self.strategy[-2])
                self.erase_lower_number = int(self.strategy[-1])
                print(f'erase part upper:{self.erase_upper_number}, lower:{self.erase_lower_number}')
            else:
                self.erase_part_number = int(self.strategy[-1])
                print(f'erase part : {self.erase_part_number}')

    def forward(self, x, seg_label):
        beta = 1.0
        # pdb.set_trace()
        if self.training:
            BC, C, H, W = x.shape
            # pdb.set_trace()
            if H != seg_label.shape[1]:
                seg_label = seg_label.to(torch.float)
                seg_label = F.interpolate(seg_label.unsqueeze(1), size=(H, W))  # BC*1*14*14

            # binary mask first
            binary_mask = x.new_ones(seg_label.size())
            # pdb.set_trace()
            if 'part' in self.strategy:
                erase_part = []
                if hasattr(self, 'erase_part_number'):
                    if self.erase_part_number == 1:
                        erase_part.append(random.randint(1, seg_label.max()))
                elif hasattr(self, 'upper_torso'):
                    erase_part.extend(random.sample(self.upper_torso, self.erase_upper_number))
                    erase_part.extend(random.sample(self.lower_torso, self.erase_lower_number))
                # erase
                for part in erase_part:
                    binary_mask[seg_label == part] = 0
            beta = torch.sigmoid(torch.sum(binary_mask) / (BC * H * W))
            # feature_map BC*1024*14*14
            x = x * binary_mask
        return x, beta
    
class MaskHead(nn.Module):
    def __init__(self, mask_dim, mask_branch, BN):
        super(MaskHead, self).__init__()
        print('using {} mask branch'.format(mask_branch))
        # add crop branch network sturctures:
        self.bottle = Bottleneck(1024, 256)
        if 'dim_red' in mask_branch:
            print('Using dim reduce for mask branch!')
            self.dim_reducer = nn.Conv2d(256 * Bottleneck.expansion, mask_dim, kernel_size=1, bias=False)
            self.dim_red = True
        else:
            self.dim_red = False
        self.mask_bn = BN(mask_dim)
        self.mask_bn.bias.requires_grad_(False)
        self.mask_bn.apply(weights_init_kaiming)
        self.mask_branch = mask_branch
        if mask_branch == 'bfe':
            self.mask = DoubleBatchDrop(h_ratio=0.1, w_ratio=1.0)  # should change
        else:
            self.mask = SegmentBatchDrop(mask_branch)
            
    def forward(self, x, seg_mask):
        x = self.bottle(x)
        beta = 1.0
        if self.mask_branch == 'bfe':
            x = self.mask(x)
        else:
            x, beta = self.mask(x, seg_mask)
        x = F.max_pool2d(x, x.size()[2:])
        if self.dim_red:
            x = self.dim_reducer(x)
        x = x.view(x.size(0), -1)
        x_ori = x
        x = self.mask_bn(x)
        return x, x_ori, beta
    
