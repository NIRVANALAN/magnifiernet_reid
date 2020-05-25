# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import numpy as np

def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def mt_collate_fn(batch):
    image, attr_label, id_label, seg, _, _ = zip(*batch)  # tuple returned
    data = torch.stack(image, 0)
    seg = torch.stack(seg, 0)
    id_label = torch.LongTensor(id_label)

    return data, id_label, seg, attr_label

def val_collate_fn(batch):
    imgs, pids, camids, im_paths = zip(*batch)
    #np.save('im.npy', torch.stack(imgs, dim=0).numpy())
    return torch.stack(imgs, dim=0), pids, camids
