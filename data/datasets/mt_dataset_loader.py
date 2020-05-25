# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import pdb
import cv2
import torchvision.transforms as T
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import random
import numpy as np
from ..transforms import RandomErasing #RandomErasing2
from .import_Market1501Attribute import import_Market1501Attribute_binary
from .import_DukeMTMCAttribute import import_DukeMTMCAttribute_binary



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class MTImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None, cfg=None):
        try:
            self.dataset, self.label2pid = dataset
        except:
            self.dataset = dataset
        self.dataset_name = cfg.DATASETS.NAMES
        if cfg.DATASETS.NAMES == 'market1501':
            train_attr, test_attr, self.attr_label = import_Market1501Attribute_binary(cfg.DATASETS.ROOT_DIR)
        elif cfg.DATASETS.NAMES == 'dukemtmc':
            train_attr, test_attr, self.attr_label = import_DukeMTMCAttribute_binary(cfg.DATASETS.ROOT_DIR)
        elif 'cuhk03' in cfg.DATASETS.NAMES:
            train_attr = []
            self.attr_label = ['not used']
        else:
            raise ValueError(f'dataset not support: {cfg.DATASETS.NAMES}')
        self.train_attr = train_attr
        self.transform = transform
        self.img_size = cfg.INPUT.SIZE_TRAIN
        self.resize = T.Resize(self.img_size)
        self.flip = T.RandomHorizontalFlip(p=1.0)
        self.crop_loc_generator = T.RandomCrop(self.img_size)
        self.pad = T.Pad(cfg.INPUT.PADDING)
        self.erase = RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        self.flip_prob = cfg.INPUT.PROB
        if self.img_size[0] == 224:
            self.seg_height = 28
            self.seg_weight = 28
        elif self.img_size[0] == 256:
            self.seg_height = 16
            self.seg_weight = 32
        elif self.img_size[0] == 384:
            if self.img_size[1] == 128:
                self.seg_height = 16
            elif self.img_size[1] == 192:
                self.seg_height = 24
            self.seg_weight = 48
    def __len__(self):
        return len(self.dataset)
    
    def attr_labels(self):
        return self.attr_label

    def num_attr_label(self):
        return len(self.attr_label)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        #if self.dataset_name == 'market1501':
        temp_root, temp_name = img_path.split('bounding')
        seg_path = temp_root + 'segmentation/' + 'bounding'+ temp_name + '.npy'
        #else:
        #    temp_root, temp_name = img_path.split('cuhk03_')
        #    seg_path = temp_root + 'segmentation/' + 'cuhk03_'+ temp_name + '.npy'
        img = read_image(img_path)
        seg = np.load(seg_path)
        seg = Image.fromarray(seg.astype('uint8')).convert('L')
        try:
            attr = np.asarray(self.train_attr[f'{self.label2pid[pid]:04}'], dtype=np.float32)
        except:
            attr = [0] #np.array(self.train_attr)
        if self.transform is not None:
            # resize
            img = self.resize(img)
            seg = self.resize(seg)
            # random horizontal flip
            if(random.random()>self.flip_prob):
                img = self.flip(img)
                seg = self.flip(seg)
            # pad
            img = self.pad(img)
            seg = self.pad(seg)
            # random crop
            crop_loc = self.crop_loc_generator.get_params(img, self.img_size)
            img = T.functional.crop(img, crop_loc[0], crop_loc[1], crop_loc[2], crop_loc[3])
            seg = T.functional.crop(seg, crop_loc[0], crop_loc[1], crop_loc[2], crop_loc[3])
            # visualization
            #img.save('/mnt/lustre/liuyuan1/cvpr20/network/MT-Net/data_quality/img/img{}.jpg'.format(index))
            #seg.save('/mnt/lustre/liuyuan1/cvpr20/network/MT-Net/data_quality/seg/seg{}.jpg'.format(index))
            # normalize and erase, only for img, not for seg
            img = self.transform(img)
            img = self.erase(img)
            #img, seg  = self.erase(img, np.array(seg))
            #temp = Image.fromarray(seg)
            #temp.save('/mnt/lustre/liuyuan1/cvpr20/network/MT-Net/data_quality/seg/seg_erase{}.jpg'.format(index))            
          
        seg = np.array(seg)
        seg = torch.from_numpy(cv2.resize(
            seg, (self.seg_height, self.seg_weight), cv2.INTER_NEAREST)).long()

        return img, pid, seg, attr, camid, img_path
