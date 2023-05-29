# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class ProbGaussianLoss(nn.Module):
    def __init__(self,
                 use_target_weight=False,
                 loss_weight = 1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
    
    def criterion(self, pred, target):
        """Criterion of Gaussian NLL Loss.

        Note:
            batch_size: N
            num_keypoints: K
        
        Args:
            pred (torch.Tensor[NxKx6]) Predicted statistics & flags.
            target (torch.Tensor[NxKx3]) Target positions & flags.
        """
    
        pose = pred[:,:,0:2]
        cov_idx = torch.tensor([[2,4],[4,3]])
        cov_mat = pred[:,:,cov_idx]
        labeled = pred[:,:,5]
        
        gt_pose = target[:,:,0:2]
        mask = (target[:,:,2] != 0)
        label_loss = torch.log(1 - labeled)
        label_loss[mask] = torch.log(labeled[mask])
        label_loss = -torch.sum(label_loss, dim=-1)

        dif = torch.reshape(gt_pose - pose, (pose.shape[0], pose.shape[1], pose.shape[2], 1))
        q = torch.matmul(torch.transpose(dif,-1,-2), torch.matmul(torch.inverse(cov_mat), dif))
        q = q.view(q.shape[0], q.shape[1])
        return torch.sum(mask * (torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455, dim=(-1)) + label_loss
    
    def forward(self, output, target, target_weight):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[NxKx6]): Output statistics & flags.
            target (torch.Tensor[NxKx3]): Target positions & flags.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output, torch.cat((target, target_weight[:,:,0].unsqueeze(-1)), dim=2))
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight