import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Diverse_Loss(object):
    def __init__(self):
        pass
    def __call__(self, feat_list):
        # feat_list is a list of bs x dim tensors
        total_sim = 0
        count = 0
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for i in range(len(feat_list)): 
            for j in range(i+1, len(feat_list)):
                batch_cos_sim = cos(F.normalize(feat_list[i]), F.normalize(feat_list[j]))
                #print('batch cos sim for pair {} and {} : {}'.format(i, j, batch_cos_sim))
                batch_cos_sim = torch.sum(batch_cos_sim) / float(batch_cos_sim.size(0)) # normalize batch wise
                total_sim += batch_cos_sim
                count += 1
        total_sim = total_sim / float(count)
        if total_sim > 0:
            return total_sim 
        else:
            return torch.Tensor([0]).cuda().squeeze()
        
                
