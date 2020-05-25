import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

from .backbones.resnet import *
import numpy as np

class WeightedAvgPooling(nn.Module):
	def __init__(self, num_ftrs=2048):
		super().__init__()
		self.num_ftrs = num_ftrs
		part_detector_block = []
		# 1*1 conv layer
		part_detector_block += [nn.Conv2d(self.num_ftrs, self.num_ftrs, 1)]
		part_detector_block += [nn.Sigmoid()]
		part_detector_block = nn.Sequential(*part_detector_block)
		part_detector_block.apply(weights_init_kaiming)
		self.part_detector_block = part_detector_block

	def forward(self, x):
		mask = self.part_detector_block(x)
		mask = torch.sum(mask * x, dim=(3, 2)) / \
			   (x.shape[-2] * x.shape[-1])  # 32 * 2048
		return mask
    
def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		init.normal_(m.weight, std=0.001)
		init.constant_(m.bias, 0)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight, std=0.001)
		init.constant_(m.bias, 0)
	elif classname.find('SyncBatchNorm2d') != -1:
		m.weight.data.fill_(1)
		m.bias.data.fill_(0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		init.normal_(m.weight.data, std=0.001)
		try:
			init.constant_(m.bias.data, 0.0)
		except:
			pass

class PCBSplitter(nn.Module):
	def __init__(self, num_parts):
		super(PCBSplitter, self).__init__()
		self.num_parts = num_parts
		for i in range(num_parts):
			setattr(self, 'chunk_' + str(i), nn.AdaptiveAvgPool2d([1, 1]))
	
	def forward(self, x):
		returnX = torch.FloatTensor().cuda()
		splittedTensor = torch.chunk(x, self.num_parts, dim=2)  # dim=1 here means to split along height
		chunkNum = 0
		for eachTensor in splittedTensor:
			chunk = getattr(self, 'chunk_' + str(chunkNum))
			pooledChunk = chunk(eachTensor)
			returnX = torch.cat((returnX, pooledChunk), dim=2)
			chunkNum += 1;
		return returnX


class AttrClassBlockFc(nn.Module):
	def __init__(self, input_dim, class_num=1, dropout=True, relu=True, bottleneck_plane=512, bn=None,
				 attr_bottleneck=False):
		super(AttrClassBlockFc, self).__init__()
		self.attr_pooling = WeightedAvgPooling(num_ftrs=2048)

		add_block = []
		if attr_bottleneck:
			self.bottle = Bottleneck(input_dim, bottleneck_plane)
			print('add Bottleneck in attr_class_block')
		else:
			self.bottle = nn.Identity()

		add_block += [nn.Linear(input_dim, bottleneck_plane)]  # linear
		add_block += [nn.BatchNorm1d(bottleneck_plane)]  # BN
		# add_block += [bn(num_bottleneck)]
		if relu:
			add_block += [nn.LeakyReLU(0.1)]
		if dropout:
			add_block += [nn.Dropout(p=0.5)]
		add_block = nn.Sequential(*add_block)
		add_block.apply(weights_init_kaiming)
		
		classifier = []
		classifier += [nn.Linear(bottleneck_plane, class_num)]
		classifier = nn.Sequential(*classifier)
		classifier.apply(weights_init_classifier)
		
		self.add_block = add_block
		self.classifier = classifier
	
	def forward(self, x, label=None):
		feat = self.bottle(x)
		x = self.attr_pooling(feat)
		x = x.view(x.size(0), x.size(1))
		# x = torch.unsqueeze(x, 2)
		# x = torch.unsqueeze(x, 3)
		feat = self.add_block(x)
		x = self.classifier(feat)
		# x = torch.squeeze(x)
		return x, feat
    
class AttrAttnBlockFc(nn.Module):
	def __init__(self, input_dim, class_num=1, dropout=True, relu=True, bottleneck_plane=512, bn=None, attr_bottleneck=False, wavp=False):
		super(AttrAttnBlockFc, self).__init__()
		self.masks = nn.Conv2d(2048, 4, 1)
		self.sm = nn.Sigmoid()
		self.wavp = wavp
		print(f'weighted_avp: {wavp}')
		if wavp:
			self.attr_pooling_gen = WeightedAvgPooling(num_ftrs=2048)
			self.attr_pooling_head = WeightedAvgPooling(num_ftrs=2048)
			self.attr_pooling_upper = WeightedAvgPooling(num_ftrs=2048)
			self.attr_pooling_lower = WeightedAvgPooling(num_ftrs=2048)
			self.clf_gen = ClassBlock(2048, 5, return_f=False, num_bottleneck=128) #temporarily disable return feature, will add asap
			self.clf_head = ClassBlock(2048, 2, return_f=False, num_bottleneck=128)
			self.clf_upper = ClassBlock(2048, 10, return_f=False, num_bottleneck=128)
			self.clf_lower = ClassBlock(2048, 13, return_f=False, num_bottleneck=128)
		else:
			self.clf_gen = nn.Linear(2048, 5)
			self.clf_head = nn.Linear(2048, 2)
			self.clf_upper = nn.Linear(2048, 10)
			self.clf_lower = nn.Linear(2048, 13)
		
		if attr_bottleneck:
			self.bottle = Bottleneck(input_dim, bottleneck_plane)
			print('add Bottleneck in attr_class_block')
		else:
			self.bottle = nn.Identity()

		self.clf_gen.apply(weights_init_classifier)
		self.clf_upper.apply(weights_init_classifier)
		self.clf_lower.apply(weights_init_classifier)
		self.clf_head.apply(weights_init_classifier)
	
	def forward(self, x, label=None):
		feat = self.bottle(x)
		attr_feat = feat
		masks = self.masks(feat) #bs, 1, h, w
		masks = self.sm(masks)
		mask_gen = masks[:,0].unsqueeze(1)
		mask_head = masks[:,1].unsqueeze(1)
		mask_upper = masks[:,2].unsqueeze(1)
		mask_lower = masks[:,3].unsqueeze(1)
		x_gen = x * mask_gen
		x_head = x * mask_head
		x_upper = x * mask_upper
		x_lower = x * mask_lower
		
		if self.wavp:
			x_gen = self.attr_pooling_gen(x_gen)
			x_head = self.attr_pooling_head(x_head)
			x_upper = self.attr_pooling_upper(x_upper)
			x_lower = self.attr_pooling_lower(x_lower)
		else:
			x_gen = F.avg_pool2d(x_gen, x_gen.size()[2:])
			x_head = F.avg_pool2d(x_head, x_head.size()[2:])
			x_upper = F.avg_pool2d(x_upper, x_upper.size()[2:])
			x_lower = F.avg_pool2d(x_lower, x_lower.size()[2:])

		x_gen = x_gen.view(x_gen.size(0), x_gen.size(1))
		x_head = x_head.view(x_head.size(0), x_head.size(1))
		x_upper = x_upper.view(x_upper.size(0), x_upper.size(1))
		x_lower = x_lower.view(x_lower.size(0), x_lower.size(1))

		x_gen = self.clf_gen(x_gen)
		x_head = self.clf_head(x_head)
		x_upper = self.clf_upper(x_upper)
		x_lower = self.clf_lower(x_lower)
		#now resemble the sequence
		x = torch.cat((x_gen, x_head, x_upper, x_lower), 1)
		'''
		if self.training == False:
			np.save('/mnt/lustre/liuyuan1/cvpr20/network/MT-Net/data_quality/attrmask/head.npy', mask_head)
			np.save('/mnt/lustre/liuyuan1/cvpr20/network/MT-Net/data_quality/attrmask/upper.npy', mask_upper)
			np.save('/mnt/lustre/liuyuan1/cvpr20/network/MT-Net/data_quality/attrmask/lower.npy', mask_lower)
		'''
		return x, mask_head, mask_upper, mask_lower, feat


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
	def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True,
				 return_f=False):
		super(ClassBlock, self).__init__()
		self.return_f = return_f
		add_block = []
		if linear:
			add_block += [nn.Linear(input_dim, num_bottleneck)]
		else:
			num_bottleneck = input_dim
		if bnorm:
			add_block += [nn.BatchNorm1d(num_bottleneck)]
		if relu:
			add_block += [nn.LeakyReLU(0.1)]
		if droprate > 0:
			add_block += [nn.Dropout(p=droprate)]
		add_block = nn.Sequential(*add_block)
		add_block.apply(weights_init_kaiming)

		classifier = []
		classifier += [nn.Linear(num_bottleneck, class_num)]
		classifier = nn.Sequential(*classifier)
		classifier.apply(weights_init_classifier)

		self.add_block = add_block
		self.classifier = classifier

	def forward(self, x):
		x = self.add_block(x)
		# pdb.set_trace()
		if self.return_f:
			f = x
			x = self.classifier(x)
			return x, f
		else:
			x = self.classifier(x)
			return x

class Pyramidal_Block_V2(nn.Module):
	def __init__(self, cur_level, level=6, dim=64, bn=None, num_feature=256):
		super(Pyramidal_Block_V2, self).__init__()
		self.level = level
		self.cur_level = cur_level
		self.dim = dim
		
		for i in range(self.level + 1 - self.cur_level):
			name = 'block' + str(i)
			# setattr(self, name, ClassBlock(2048, 100, False, False, num_feature, bn))
			setattr(self, name, ClassBlock(2048, 100, False, True, 128 if self.cur_level == 6 else 256, bn))
	
	def forward(self, x):
		all_part = []
		for i in range(self.level + 1 - self.cur_level):
			start = i * x.size()[2] / self.level
			end = i * x.size()[2] / self.level + self.cur_level * x.size()[2] / self.level - 1
			part = x[:, :, int(start):int(end), :]
			kernel_size = [part.size()[2], part.size()[3]]
			part = torch.nn.functional.avg_pool2d(part, kernel_size)
			part = torch.squeeze(part)
			name = 'block' + str(i)
			block = getattr(self, name)
			part = block(part)
			all_part.append(part)
		# result=all_part[0]
		# for k in all_part[1:]:
		#    result=torch.cat([result,k], dim=1)
		return torch.cat(all_part, dim=1)


class Pyramid_V2(nn.Module):
	def __init__(self, level=6, level_choose=[1, 1, 1, 1, 1, 1], group_size=1, group=1, sync_stats=False,
	             num_feature=256):
		super(Pyramid_V2, self).__init__()
		
		#def BNFunc(*args, **kwargs):
			#return SyncBatchNorm2d(*args, **kwargs, group=group, sync_stats=sync_stats, var_mode=syncbnVarMode_t.L2)
		
		BN = nn.BatchNorm2d #BNFunc
		self.level = level
		self.level_choose = level_choose
		## using res101 backbone
		model_ft = resnet101(group_size=group_size, group=group, sync_stats=sync_stats, bb_only=True)
		self.model = model_ft
		self.valid_level = list(filter(lambda x: x == 1, self.level_choose))
		self.num_classifier = len(self.valid_level)
		self.dim = int(num_feature / self.num_classifier)
		
		for i in range(self.level):
			if self.level_choose[i] == 0:
				continue
			name = 'P_Block' + str(i)
			setattr(self, name, Pyramidal_Block_V2(i + 1, self.level, self.dim, BN, num_feature))
			if i == 5:
				continue
			name = 'Dim_Reducer' + str(i)
			setattr(self, name, ClassBlock(256 * (self.level - i), 256, False, False, self.dim, BN))
	
	def forward(self, x, label=None):
		x = self.model.conv1(x)
		x = self.model.bn1(x)
		x = self.model.relu(x)
		x = self.model.maxpool(x)
		x = self.model.layer1(x)
		x = self.model.layer2(x)
		x = self.model.layer3(x)
		x = self.model.layer4(x)
		# y={}
		predict = []
		for i in range(self.level):
			if self.level_choose[i] == 0:
				continue
			name = 'P_Block' + str(i)
			block = getattr(self, name)
			part = block(x)
			if i != 5:
				name = 'Dim_Reducer' + str(i)
				reducer = getattr(self, name)
				if self.dim != 256:
					part = reducer(part)
				part = torch.squeeze(part)
			# y[i] = part
			predict.append(part)
		# for i, j in enumerate(y.values()):
		#    predict.append(j)
		return predict
