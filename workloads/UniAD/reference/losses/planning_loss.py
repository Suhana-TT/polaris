# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmdet.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    def __init__(self, loss_type="L2"):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        return torch.sum(err * mask) / (torch.sum(mask) + 1e-5)


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight

    def forward(
        self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox
    ):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(
            1,
        )
        dump_sdc = []
        for i in range(n_futures):
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[
                    :, [0, 3, 4, 7], :2
                ]  # (N, 8, 3) -> (N, 4, 2) only bev
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners(
                    [
                        sdc_traj_all[0, i, 0],
                        sdc_traj_all[0, i, 1],
                        self.w,
                        self.h,
                        sdc_yaw,
                    ]
                )
                dump_sdc.append(sdc_bev_box.cpu().detach().numpy())
                for j in range(future_gt_bbox_corners.shape[0]):
                    inter_sum += self.inter_bbox(
                        sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device)
                    )
        return inter_sum * self.weight

    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])

        xi1, yi1 = torch.minimum(xa1, xb1), torch.minimum(ya1, yb1)
        xi2, yi2 = torch.maximum(xa2, xb2), torch.maximum(ya2, yb2)
        xi1, yi1 = torch.minimum(xa1, xb1), torch.minimum(ya1, yb1)
        xi2, yi2 = torch.maximum(xa2, xb2), torch.maximum(ya2, yb2)
        w = torch.clamp(xi1 - xi2, min=0)
        h = torch.clamp(yi1 - yi2, min=0)
        intersect = w * h
        return intersect

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        
        # 4, 2 corners in the box's local coordinate frame, built from tensors to
        # preserve device, dtype, and gradients.
        corners = torch.stack(
            [
                torch.stack([w / 2, -l / 2]),
                torch.stack([w / 2, l / 2]),
                torch.stack([-w / 2, l / 2]),
                torch.stack([-w / 2, -l / 2]),
            ],
            dim=0,
        )  # shape: (4, 2)
        # 2x2 rotation matrix using tensor operations
        rot_mat = torch.stack(
            [
                torch.stack([torch.cos(theta), torch.sin(theta)]),
                torch.stack([-torch.sin(theta), torch.cos(theta)]),
            ],
            dim=0,
        )
        # Translation uses stack on existing tensors instead of torch.tensor(bbox[:2])
        translation = torch.stack([x, y])[:, None]  # shape: (2, 1)
        new_corners = rot_mat @ corners.T + translation

        
        return new_corners.T
