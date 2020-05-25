import torch.nn as nn
import math
import random
import torchvision
import torch
from torch.nn import functional as F
from torch.nn import init
import numpy as np

from ..backbones.resnet import *
from ..Tools import weights_init_kaiming

class SegmentAllPart(nn.Module):
    def __init__(self):
        super().__init__()
        self.parts = list(range(8))

    def forward(self, x, seg_label):
        BC, C, H, W = x.shape
        # pdb.set_trace()
        if H != seg_label.shape[1]:
            seg_label = seg_label.to(torch.float)
            seg_label = F.interpolate(seg_label.unsqueeze(1), size=(H, W))  # BC*1*14*14

        masks = [x.new_zeros(seg_label.size()) for _ in range(len(self.parts))]
        for i, part in enumerate(self.parts):
            masks[i][seg_label == part] = 1
        betas = []
        part_feats = []
        for m in masks:
            part_feats.append(x * m)
            betas.append(torch.sigmoid(torch.sum(m) / (BC * H * W))) 
        return part_feats, betas
    
class SegmentAttrPart(nn.Module):
    def __init__(self, attr_mask_weight):
        super().__init__()
        self.parts = list(range(8))
        self.attr_mask_weight = attr_mask_weight
        print('Attribute mask weight in part branch is {}'.format(self.attr_mask_weight))
    def forward(self, x, seg_label, head_mask, upper_mask, lower_mask):
        BC, C, H, W = x.shape
        # pdb.set_trace()
        if H != seg_label.shape[1]:
            seg_label = seg_label.to(torch.float)
            seg_label = F.interpolate(seg_label.unsqueeze(1), size=(H, W))  # BC*1*14*14
        if head_mask is not None:
            if H != head_mask.shape[2]:
                head_mask = F.interpolate(head_mask, size=(H, W))
                upper_mask = F.interpolate(upper_mask, size=(H, W))
                lower_mask = F.interpolate(lower_mask, size=(H, W))
        masks = [x.new_zeros(seg_label.size()) for _ in range(len(self.parts))]
        for i, part in enumerate(self.parts):
            if head_mask is not None:
                if i == 1:
                    masks[i] = self.attr_mask_weight * head_mask.detach() #should i detach
                elif i in [2,3,4]:
                    masks[i] = self.attr_mask_weight * upper_mask.detach()
                elif i in [5,6,7]:
                    masks[i] = self.attr_mask_weight * lower_mask.detach()           
            masks[i][seg_label == part] = 1
        betas = []
        part_feats = []
        for m in masks:
            part_feats.append(x * m)
            betas.append(torch.sigmoid(torch.sum(m) / (BC * H * W)))
        return part_feats, betas

class PartDistanceHead(nn.Module):
    def __init__(self, part_dim, BN, attr_mask_weight, part_layer):
        super(PartDistanceHead, self).__init__()
        print('using part distance branch')
        #self.dim_reducer = nn.ModuleList([nn.Conv2d(256 * Bottleneck.expansion, part_dim, \
        #                                            kernel_size=1, bias=False) for _ in range(8)])
        if part_layer == 4:
            dim = 2048
        elif part_layer == 3:
            dim = 1024
        else:
            raise ValueError('Invalid layer!')
        self.bottle = Bottleneck(dim, int(dim / 4))
        self.part_bns = nn.ModuleList([BN(dim) for _ in range(8)])
        for bn in self.part_bns:
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
        self.crop_module = SegmentAttrPart(attr_mask_weight) #SegmentAllPart() #SegmentGroupPart()
        #random.seed(0)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.combine_bn = BN(dim)
        self.combine_bn.bias.requires_grad_(False) 
        self.combine_bn.apply(weights_init_kaiming)                
    def forward(self, x, seg_mask, head_mask, upper_mask, lower_mask):
        #div_x = x
        x = self.bottle(x)
        #div_x = self.bottle(div_x.detach())
        #np.save('part_div_vis.npy', x.detach().cpu().numpy())
        #raise ValueError('Terminated by LY')
        all_parts, betas = self.crop_module(x, seg_mask, head_mask, upper_mask, lower_mask)
        x_combine_nobn = torch.Tensor([]).cuda()
        combine_parts = []
        combine_parts_nobn = []
        for i in range(len(all_parts)):
            p = all_parts[i]
            p = F.max_pool2d(p, p.size()[2:])
            #p = self.dim_reducer[i](p)
            p = p.view(p.size(0), -1)
            combine_parts_nobn.append(p)
            p = self.part_bns[i](p)
            #print(p)
            combine_parts.append(p)
            #x_combine = torch.cat((x_combine, p),1)
        # lstm
        x_combine_nobn = torch.stack(combine_parts_nobn, 1)
        _, (x_combine_nobn, _) = self.lstm(x_combine_nobn)
        #print(x_combine_nobn.size()) seq x bs x fdim
        x_combine_nobn = x_combine_nobn.squeeze()
        x_combine = self.combine_bn(x_combine_nobn)
        #return combine_parts, [x_combine, x_combine_nobn, div_feats '''combine_parts_nobn'''], betas
        return combine_parts, [x_combine, x_combine_nobn, combine_parts_nobn], betas
        # x_combine is for softmax loss, x_combine_nobn is for triplet loss, combine_parts_nobn is for div loss

class PartAblationHead(nn.Module):
    def __init__(self, part_dim, BN, attr_mask_weight, part_layer):
        super(PartAblationHead, self).__init__()
        print('using part ablation study branch')
        if part_layer == 4:
            dim = 2048
        elif part_layer == 3:
            dim = 1024
        else:
            raise ValueError('Invalid layer!')
        self.bottle = Bottleneck(dim, int(dim / 4))
        self.dim_reducer = nn.ModuleList([nn.Conv2d(int(dim / 4) * Bottleneck.expansion, part_dim, \
                                                    kernel_size=1, bias=False) for _ in range(8)])
        self.part_bns = nn.ModuleList([BN(part_dim) for _ in range(8)])
        for bn in self.part_bns:
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
        self.crop_module = SegmentAttrPart(attr_mask_weight) #SegmentAllPart() #SegmentGroupPart()
            # set random seed, very important, it help to sync all processes
        random.seed(0)
        self.combine_bn = BN(dim)
        self.combine_bn.bias.requires_grad_(False)
        self.combine_bn.apply(weights_init_kaiming)
    def forward(self, x, seg_mask, head_mask, upper_mask, lower_mask):
        x = self.bottle(x)
        #np.save('part_div_vis.npy', x.detach().cpu().numpy()) 
        #raise ValueError('Terminated by LY')    
        all_parts, betas = self.crop_module(x, seg_mask, head_mask, upper_mask, lower_mask)
        x_combine_nobn = torch.Tensor([]).cuda()
        combine_parts = []
        combine_parts_nobn = []
        for i in range(len(all_parts)):
            p = all_parts[i]
            p = F.max_pool2d(p, p.size()[2:])
            p = self.dim_reducer[i](p)
            p = p.view(p.size(0), -1)
            combine_parts_nobn.append(p)
            p = self.part_bns[i](p)
            #print(p)
            combine_parts.append(p)
        # lstm
        x_combine_nobn = torch.cat(tuple(combine_parts_nobn), 1)
        x_combine = self.combine_bn(x_combine_nobn)
        #return combine_parts, [x_combine, x_combine_nobn, div_feats '''combine_parts_nobn'''], betas
        return combine_parts, [x_combine, x_combine_nobn, combine_parts_nobn], betas
