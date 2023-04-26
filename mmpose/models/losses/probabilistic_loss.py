# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES

@LOSSES.register_module()
class GaussianNLLLoss(nn.Module):
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
            loss = self.criterion(output * target_weight.unsqueeze(-1),
                                  target * target_weight.unsqueeze(-1))
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight

@LOSSES.register_module()
class GaussianNLLHeatmapLoss(nn.Module):
    def __init__(self,
                 use_target_weight=False,
                 loss_weight=1.0):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight
    
    def criterion(self, pred, target):
        n, c, h, w = pred.shape
        max_ = torch.max(torch.max(pred, dim=-1)[0], dim=-1, keepdim=True)[0].unsqueeze(-1)
        exp_max_ = torch.exp(max_ - h - w)
        z = torch.sum(torch.exp(pred - max_), (2, 3)).view(n, c, 1, 1)
        h_norm = torch.exp(pred - max_) / z

        x_vals = torch.linspace(0, w-1, w, device=h_norm.device).unsqueeze(0)
        y_vals = torch.linspace(0, h-1, h, device=h_norm.device).unsqueeze(1)

        x_means = torch.sum(h_norm * x_vals, dim = (2, 3))
        y_means = torch.sum(h_norm * y_vals, dim = (2, 3))
        
        xn = (x_vals - x_means.view(n, c, 1, 1))
        yn = (y_vals - y_means.view(n, c, 1, 1))

        x_var = 1/12 + torch.sum(h_norm * xn * xn, dim=(2,3))
        y_var = 1/12 + torch.sum(h_norm * yn * yn, dim=(2,3))
        xy_covar = torch.sum(h_norm * xn * yn, dim=(2,3))
        presence_prob = 1 - 1 / (exp_max_ * z + 1).view(n, c)

        pred = torch.stack((x_means, y_means, x_var, y_var, xy_covar, presence_prob), -1)
    
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
        if self.use_target_weight:
            loss = self.criterion(output * target_weight.unsqueeze(-1),
                                  target * target_weight.unsqueeze(-1))
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight
