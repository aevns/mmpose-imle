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
        pose_loss = torch.sum(mask * ((torch.log(torch.det(cov_mat)) + q)/2 + 1.8378770664093455), dim=(-1))
        # Logs are taken from (0.0001, 1] for numerical stability
        # This should ensure wildly wrong predictions don't have exploding gradients
        if self.use_label_loss:
            label_loss = -torch.log(1 - labeled * (1 - 1E-4))
            label_loss[mask] = -torch.log(1E-4 + labeled[mask] * (1 - 1E-4))
            pose_loss += torch.sum(label_loss, dim=-1)
        assert(~(torch.isnan(pose_loss).any()))
        return pose_loss * self.loss_weight
    
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
                target_weights: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        B, K, H, W = output.shape
        dist = torch.log_softmax(output.view(B, K, H * W), dim=(2)).view(B, K, H, W)
        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.kl_div(dist, target, log_target=True)
        else:
            _loss = F.kl_div(dist, target, log_target=True, reduction='none')
            loss = torch.sum(_loss * _mask)
        
        assert(~(torch.isnan(loss).any()))
        if self.use_label_loss:
            labeled =  1 - 1 / (torch.sum(torch.exp(output), dim=(2, 3)) + 1)
            label_loss = -torch.log(1 - labeled * (1 - 1E-4))
            label_loss[mask] = -torch.log(1E-4 + labeled[mask] * (1 - 1E-4))
            assert(~(torch.isnan(label_loss).any()))
            loss += torch.sum(label_loss)
        return loss * self.loss_weight

    def _get_mask(self, target: torch.Tensor, target_weights: Optional[torch.Tensor],
                  mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask