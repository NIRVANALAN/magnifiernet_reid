import torch

class PSLoss(object):

    def __init__(self, normalize_size=True, num_classes=8):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none' if normalize_size else 'mean')
        self.normalize_size = normalize_size
        self.num_classes = num_classes

    def __call__(self, pred, label):

        # Calculation
        ps_pred = pred
        ps_label = label
        N, C, H, W = ps_pred.size()
        assert ps_label.size() == (N, H, W)

        # shape [N, C, H, W] -> [N, H, W, C] -> [NHW, C]
        ps_pred = ps_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # shape [N, H, W] -> [NHW]
        ps_label = ps_label.view(N * H * W).detach()
        loss = self.criterion(ps_pred, ps_label)
        if self.normalize_size:
            loss_ = 0
            cur_batch_n_classes = 0

            for i in range(self.num_classes):
                loss_i = loss[ps_label == i]
                if loss_i.numel() > 0:
                    loss_ += loss_i.mean()
                    cur_batch_n_classes += 1
            loss_ /= (cur_batch_n_classes + 1e-8)
            loss = loss_

        return loss
