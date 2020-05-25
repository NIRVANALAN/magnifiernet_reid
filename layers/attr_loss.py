import torch
import torch.nn as nn
import numpy as np


class Attr_Loss(object):
    def __init__(self, factor=8):
        self.attr_criterion = torch.nn.KLDivLoss().cuda()
        self.factor = factor

    def __call__(self, pred, labels):
        # convert to cuda
        for i in range(len(labels)):
            labels[i] = labels[i].cuda(async=True)
            # calculate attributes loss
        attr_current_num = 1
        loss_attr = self.attr_criterion(pred[0], labels[0])
        if loss_attr.data < 0:
            print('attribute loss negative: attr_index_{}:{}'.format(
                0, loss_attr.data))

        for i in range(1, 16):  #
            attr_current_num += 1
            _loss = self.attr_criterion(pred[i], labels[i])
            if _loss.data < 0:
                print('attribute loss negative: attr_index_{}:{}'.format(
                    i, _loss.data))
            loss_attr += _loss

        loss_attr = loss_attr / attr_current_num / self.factor
        return loss_attr

class AttributeLoss(object):
	def __init__(self, factor=8):
		self.attr_criterion = nn.BCEWithLogitsLoss().cuda()
		if factor:
			self.factor = 1 / factor
		else:
			self.factor = 0
		print(f'APR: {self.factor}')

	def __call__(self, pred, labels):
		# convert to cuda
		loss_attr = self.attr_criterion(pred, labels)

		loss_attr = loss_attr * self.factor
		return loss_attr

