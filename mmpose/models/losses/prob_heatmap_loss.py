# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class ProbHeatmapLoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight=False, loss_weight=1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        reduction = 'none' if use_target_weight else 'batchmean'
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.loss_weight = loss_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        norms_gt = torch.sum(heatmaps_gt, dim=(2,3))
        heatmaps_gt = heatmaps_gt / norms_gt
        norms_pred = torch.sum(torch.exp(heatmaps_pred), dim=(2,3))
        heatmaps_pred = heatmaps_pred - torch.log(norms_gt)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss_joint = self.criterion(heatmap_pred, heatmap_gt)
                loss_joint = loss_joint * target_weight[:, idx]
                loss += loss_joint.mean()
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints * self.loss_weight