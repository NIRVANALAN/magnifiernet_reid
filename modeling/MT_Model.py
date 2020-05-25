from __future__ import absolute_import

#import pdb
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Tools import *
from .MT_Net import MTNet

class MTModel(nn.Module):
    def __init__(self,
                 test=False,
                 num_features=256,
                 dropout=0.7,
                 num_classes=0,
                 last_stride=1,
                 backbone_name='resnet50',
                 attr_pooling='embedded',
                 in_channels=2048,
                 mid_c=256,
                 num_classes_seg=8,
                 global_branch=True,
                 mask_branch=False, 
                 part_branch=False, 
                 mask_dim=512, 
                 part_dim=128, 
                 attr_label_number=28,
                 part_info=None,
                 attr_mask_weight=0.5,
                 wavp=False,
                 use_attr=True,
                 part_layer=4,
                 part_abla=False):
        super(MTModel, self).__init__()
        #if arch not in FACTORY:
        #    raise KeyError("Unknown models: ", arch)

        print('create %s model with %d' % (backbone_name, num_features))
        print('Global Branch: {}, Mask Branch: {}, Part Branch: {}'.format(global_branch, mask_branch, part_branch))

        self.base = MTNet(
            backbone_name=backbone_name,
            last_conv_stride=last_stride,
            in_channels=in_channels,
            mid_c=mid_c,
            num_classes=num_classes_seg,
            num_features=num_features,
            global_branch=global_branch,
            mask_branch=mask_branch, 
            part_branch=part_branch, 
            mask_dim=mask_dim, 
            part_dim=part_dim, 
            part_info=part_info,
            attr_label_number=attr_label_number,
            attr_mask_weight=attr_mask_weight,
            wavp=wavp,
            use_attr=use_attr,
            part_layer=part_layer,
            part_abla=part_abla)  # B,C
            
        self.dropout = dropout
        self.num_features = num_features
        self.num_classes = num_classes
        self.test = test
        self.global_branch = global_branch
        self.mask_branch = mask_branch
        self.part_branch = part_branch

        if self.dropout > 0:
            self.drop = nn.Dropout(dropout)
        if self.num_classes > 0:
            if self.global_branch:
                self.classifier = nn.Linear(num_features, num_classes)
                self.classifier.apply(weights_init_classifier)
            # mask
            if mask_branch:
                self.classifier_mask = nn.Linear(mask_dim, num_classes)
                self.classifier_mask.apply(weights_init_classifier)
            if part_branch:
                n_parts = 8
                if part_abla:
                   mult = n_parts
                else:
                   mult = 1
                self.classifier_part = nn.ModuleList([nn.Linear(part_dim, \
                                        num_classes) for _ in range(n_parts)]) 
                for clf in self.classifier_part:
                   clf.apply(weights_init_classifier)
                self.classifier_anchor = nn.Linear(int(part_dim*mult),\
                                                       num_classes)
                self.classifier_anchor.apply(weights_init_classifier)
                
    def load_param(self, trained_path):
        try:
            param_dict = torch.load(trained_path).state_dict()
        except:
            param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            try:
                self.state_dict()[i].copy_(param_dict[i])
            except:
                print('missing module {} in current model'.format(i))

    def forward(self, x, seg_label=None, id_label=None):
        gx, seg_result, attr_res, attr_feat, mx, mbeta, px, p_Schrödinger, pbetas, gx_ori, mx_ori = self.base(x, seg_label)
        panchor = None
        if self.part_branch:
            panchor = p_Schrödinger[0]

        # here we save a copy of all branch features for triplet loss
        glb_feat = gx_ori
        mask_feat = mx_ori
        part_div_feat = px
       # part_feat = px part feature before bn is probably in the index1 at p_schrodinger
 
        if self.dropout > 0:
            if self.global_branch:
                gx = self.drop(gx)
            if self.mask_branch:
                mx = self.drop(mx)
            if self.part_branch:
                for i in range(len(px)):
                    px[i] = self.drop(px[i])
                panchor = self.drop(panchor)

        if self.training == False: #self.test: actually can delete self,test already, todo
            feats = [gx, mx, panchor]
            #feats = [attr_feat]
            #feats = [panchor]
            test_out = torch.Tensor([]).cuda()
            for f in feats:
                if f is not None:
                    test_out = torch.cat((test_out, f), 1)
            return test_out
        
        if self.num_classes > 0:
            if self.global_branch:
                gx = self.classifier(gx)
            if self.mask_branch:
                mx = self.classifier_mask(mx)
            if self.part_branch:
                px_temp = [] #cant do in-place cuz part_div_feat is reference it as well
                for i in range(len(px)):
                    px_temp.append(self.classifier_part[i](px[i]))
                px = px_temp
                panchor = self.classifier_anchor(panchor)
        return [gx, seg_result, attr_res, mx, mbeta, px, panchor, pbetas, p_Schrödinger, glb_feat, mask_feat, part_div_feat]
