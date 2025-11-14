# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

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
        pose_loss = torch.sum(mask * ((torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455), dim=(-1))
        # Logs are taken from (0.00001, 1] for numerical stability
        # This ensures wildly wrong predictions don't have exploding gradients
        if self.use_label_loss:
            label_loss = -torch.log(1 - labeled * (1 - 1E-5))
            label_loss[mask] = -torch.log(1E-5 + labeled[mask] * (1 - 1E-5))
            pose_loss += torch.sum(label_loss, dim=-1)
    
        return pose_loss * self.loss_weight