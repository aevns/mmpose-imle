# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class ProbHeatmapLoss(nn.Module):
    
    @staticmethod
    def KLDivLoss(reduction = 'none', eps = 1E-6):
        if reduction == 'none':
            def loss(q,p):
                return p * (torch.log(p + eps) - q)
        elif reduction == 'batchmean':
            def loss(q,p):
                #batchsize = q.size(0)
                return torch.sum(p * (torch.log(p + eps) - q))#/batchsize
            
        return loss

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        reduction = 'none' if use_target_weight else 'batchmean'
        self.criterion = ProbHeatmapLoss.KLDivLoss(reduction=reduction)
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight, use_label_loss = False):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        n, c, h, w = output.shape
        with torch.no_grad():
            max_ = torch.max(torch.max(output, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
        z = torch.sum(torch.exp(output - max_), (2, 3)).view(n, c, 1, 1)
        output = output - max_ - torch.log(z)

        presence_prob = 1 - 1 / (torch.exp(max_) * z / (h*w) + 1).view(n, c)
        mask = (target_weight[:,:,0] != 0)
        label_loss = -torch.log(1 - presence_prob * (1 - 1E-4))
        label_loss[mask] = -torch.log(presence_prob[mask] * (1 - 1E-4))

        with torch.no_grad():
            target_min = torch.min(torch.min(target, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
            target = target - target_min
            norms_target = torch.sum(target, dim=(2,3), keepdim=True)
            target = target / norms_target
            target[~mask] = 1. / (w*h)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        if ~use_label_loss:
            label_loss *= 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss_joint = self.criterion(heatmap_pred, heatmap_gt)
                loss_joint[~mask[:,idx]] = 0
                loss_joint = loss_joint * target_weight[:, idx]
                loss += (loss_joint.sum() + label_loss[:, idx].sum()) / batch_size
            else:
                loss_stage = self.criterion(heatmap_pred, heatmap_gt)
                loss_stage[~mask[:,idx]] = 0
                loss += loss_stage + label_loss[:, idx] / batch_size

        return loss * self.loss_weight