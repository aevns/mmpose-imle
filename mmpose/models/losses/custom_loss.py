# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.registry import MODELS


@MODELS.register_module()
class NLLGaussianLoss(nn.Module):

    def __init__(self,
                 use_target_weight=False,
                 use_label_loss=True,
                 loss_weight=1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.use_label_loss = use_label_loss
        self.loss_weight = loss_weight
        self.cov_idx = torch.tensor([[2,4],[4,3]])
    
    def forward(self, output, target, target_weight = None):

        pose = output[:,:,0:2]
        cov_mat = output[:,:,self.cov_idx]
        labeled = output[:,:,5]
        
        gt_pose = target[:,:,0:2]
        mask = (target_weight > 0.5)
        dif = torch.reshape(gt_pose - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
        q = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(cov_mat), dif))
        q = q.view(q.shape[0], q.shape[1])
        #pose_loss = torch.sum(mask * ((torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455), dim=(-1))
        pose_loss = torch.sum(mask * ((torch.log(torch.det(cov_mat)) + q)/2 + 10.514990744), dim=(-1))
        # Logs are taken from (0.0001, 1] for numerical stability
        # This should ensure wildly wrong predictions don't have exploding gradients
        if self.use_label_loss:
            label_loss = -torch.log(1 - labeled * (1 - 1E-4))
            label_loss[mask] = -torch.log(1E-4 + labeled[mask] * (1 - 1E-4))
            pose_loss += torch.sum(label_loss, dim=-1)
        return torch.sum(pose_loss) * self.loss_weight
    
@MODELS.register_module()
class DKLHeatmapLoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 use_label_loss: bool = True,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.use_label_loss = use_label_loss
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(self,
                output: torch.Tensor,
                target: torch.Tensor,
                target_weights: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, K, H, W = output.shape
        dist = torch.log_softmax(output.view(B, K, H * W), dim=(2)).view(B, K, H, W)
        loss = F.kl_div(dist, target, log_target=True, reduction='none').sum(dim=(2, 3))
        mask = (target_weights > 0.5)
        loss = torch.sum(loss * mask, dim=(-1))
        
        if self.use_label_loss:
            labeled =  1 - 1 / (torch.sum(torch.exp(output), dim=(2, 3)) + 1)
            label_loss = -torch.log(1 - labeled * (1 - 1E-4))
            label_loss[mask] = -torch.log(1E-4 + labeled[mask] * (1 - 1E-4))
            loss += torch.sum(label_loss, dim=(-1))
        return torch.sum(loss) * self.loss_weight