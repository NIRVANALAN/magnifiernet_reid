import torch
#import pdb
import random
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

#from .resnet import *
from .backbones.resnet import *
from .Tools import weights_init_kaiming, AttrClassBlockFc, AttrAttnBlockFc
from .modules.MaskModule import *
from .modules.PartModule import *
from .modules.SegAttrModule import *

CR_FACTORY = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
}

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class MTNet(nn.Module):
    def __init__(self, 
                 backbone_name='resnet50',
                 last_conv_stride=1, 
                 in_channels=2048, 
                 mid_c=256, 
                 num_classes=8,
                 num_features=256, 
                 global_branch=True,
                 mask_branch=False, 
                 part_branch=False, 
                 mask_dim=512, 
                 part_dim=128, 
                 part_info=None,
                 pretrain_choice=True,
                 attr_bottleneck_plane=512,
                 attr_label_number=28,
                 attr_feature_bottleneck=True,
                 attr_mask_weight=0.5,
                 seg_tune=False,
                 wavp=False,
                 use_attr=True,
                 part_layer=4,
                 part_abla = False
                ):

        super(MTNet, self).__init__()

        #def BNFunc(*args, **kwargs):
        #    return SyncBatchNorm2d(*args, **kwargs, group=group, sync_stats=sync_stats, var_mode=syncbnVarMode_t.L2)
        BN = nn.BatchNorm1d
        # using res101 backbone
        model_ft = CR_FACTORY[backbone_name](last_stride=last_conv_stride)
        # print('Backbone arch: {}'.format(backbone_name))
        self.model = model_ft
        self.global_branch = global_branch
        self.attr_label_number = attr_label_number
        self.use_attr = use_attr
        self.part_layer = part_layer
        if self.global_branch:
       #     self.feat = nn.Conv2d(512 * Bottleneck.expansion, num_features, kernel_size=1, bias=False)
            self.feat_bn = BN(num_features)
            self.feat_bn.bias.requires_grad_(False)  # freeze gamma
            self.feat_bn.apply(weights_init_kaiming)
        
        # attribute task
        if self.use_attr:
            print('Using attributes\n')
            self.__setattr__('class_attr', AttrAttnBlockFc(input_dim=2048, class_num=self.attr_label_number, \
                                                bottleneck_plane=attr_bottleneck_plane, \
                                                        attr_bottleneck=attr_feature_bottleneck, wavp=wavp))
        # mask task
        self.mask_branch = mask_branch
        if mask_branch:
            mask_strategy = mask_branch
            self.mask = MaskHead(mask_dim, mask_strategy, BN)   
        # part task
        self.part_branch= part_branch 
        if part_branch:
            if part_abla:
                print('Part Abaltion Study')
                self.part = PartAblationHead(part_dim, BN, attr_mask_weight, part_layer)
            else:
                self.part = PartDistanceHead(part_dim, BN, attr_mask_weight, part_layer)
        # segmentation task    
        self.seg_module = PartSegModule(in_channels, mid_c, num_classes)
        self.seg_tune = seg_tune
        if(self.seg_tune):
            print('Using seg prediction to fine-tune...')
        if pretrain_choice:
            try:
                self.model.load_param('/mnt/lustre/liuyuan1/.torch/models/resnet50-19c8e357.pth')
                print('Loading pretrained ImageNet model......')
            except:
                print('pre-train model not found or struct does not fit')
                pass
    def forward(self, x, seg_label=None):
        # backbone features
        #x_vis = x.detach().cpu().numpy()
        x, aux = self.model(x)
        # visualization
        #x_vis2 = aux.detach().cpu().numpy()
        #np.save('topk_im-best-.npy', x_vis)
        #np.save('topk_act-best-.npy', x_vis2)
        #raise ValueError('visualization Force Terminated..')
        # attr branch
        attr_feat = None
        head_mask = None
        upper_mask = None
        lower_mask = None
        attr_res = None
        if self.use_attr:
            attr_res, head_mask, upper_mask, lower_mask, attr_feat = self.__getattr__('class_attr')(x)
            attr_feat = F.avg_pool2d(attr_feat, attr_feat.size()[2:]).view(attr_feat.size(0),-1)
        #print('running')
        #attr_res, attr_feat = self.__getattr__('class_attr')(x)
        # seg branch
        #print(x.size()) 14 x 14 for 224 x 224, 16 x 8 for 256 x 128
        seg_result = self.seg_module(x)
       # print(seg_result.size()) 28 x 28 for 224, 32 x 16 for 256
        # global branch 
        gx = None
        gx_ori = None
        if self.global_branch:
            gx = F.avg_pool2d(x, x.size()[2:])
        #    gx = self.feat(gx)
            gx = gx.view(gx.size(0),-1)
            gx_ori = gx
            gx = self.feat_bn(gx)
        
        # mask branch
        mx = None
        mx_ori = None
        mbeta = None
        if self.mask_branch:
            mx, mx_ori, mbeta = self.mask(aux, seg_label)
        # part branch
        pbetas = None
        px = None
        p_Scrodinger = None 
        if self.part_branch:
            if self.part_layer == 3:
                x = aux
            if self.training == False or self.seg_tune:
                seg_pred = SegPred(seg_result)               
                px, p_Scrodinger, pbetas = self.part(x, seg_pred.detach(), head_mask, upper_mask, lower_mask)
            elif self.training == True and self.seg_tune == False:
                px, p_Scrodinger, pbetas = self.part(x, seg_label, head_mask, upper_mask, lower_mask)
            else:
                raise ValueError('Seg Tune got problem!')
        return gx, seg_result, attr_res, attr_feat, mx, mbeta, px, p_Scrodinger, pbetas, gx_ori, mx_ori
               #global #seg #attr #mask_feat #mask_beta #part_feats #part_anchor_feat #part_betas
