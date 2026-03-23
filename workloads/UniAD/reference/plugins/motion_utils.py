# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

import numpy as np
import random
from .motion_optimization import MotionNonlinearSmoother


def nonlinear_smoother(gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, bbox_tensor):
    """
    Applies nonlinear smoother to ground truth future trajectories.
    All inputs are numpy arrays (no torch tensors).
    """
    gt_bboxes_3d = np.asarray(gt_bboxes_3d)
    gt_fut_traj = np.asarray(gt_fut_traj)
    gt_fut_traj_xy_diff = np.zeros((gt_fut_traj.shape[0], 13, 2))
    gt_fut_traj_xy_diff[:, 1:, :] = gt_fut_traj
    gt_fut_traj_xy_diff = np.diff(gt_fut_traj_xy_diff, axis=1)
    gt_fut_traj_yaw = np.arctan2(
        gt_fut_traj_xy_diff[:, :, 1], gt_fut_traj_xy_diff[:, :, 0]
    )
    gt_fut_traj_yaw = np.concatenate(
        [gt_bboxes_3d[:, None, 6:7], gt_fut_traj_yaw[:, :, None]], axis=1
    )
    gt_fut_traj = np.concatenate([gt_bboxes_3d[:, None, :2], gt_fut_traj], axis=1)

    gt_fut_traj_mask = np.asarray(gt_fut_traj_mask)
    bbox_tensor = np.asarray(bbox_tensor)
    ts_limit = gt_fut_traj_mask.sum(1)[:, 0]
    yaw_preds = bbox_tensor[:, 6]
    vel_preds = bbox_tensor[:, -2:]
    speed_preds = np.sqrt(np.sum(vel_preds**2, axis=-1))
    traj_perturb_all = []

    def _is_dynamic(traj, ts, dist_thres):
        return np.sqrt(np.sum((traj[ts, :2] - traj[0, :2]) ** 2)) > dist_thres

    def _check_diff(x_curr, ref_traj):
        if (
            np.sqrt(
                (x_curr[0] - ref_traj[0, 0]) ** 2 + (x_curr[1] - ref_traj[0, 1]) ** 2
            )
            > 2
        ):
            return False
        a = np.array([np.cos(x_curr[2]), np.sin(x_curr[2])])
        b = np.array([np.cos(ref_traj[0, 2]), np.sin(ref_traj[0, 2])])
        diff_theta = np.arccos(
            np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
        )
        if diff_theta > np.pi / 180 * 30:
            return False
        return True

    def _check_ade(traj_pert, traj_ref, thres):
        return (
            np.mean(np.sqrt(np.sum((traj_pert[:, :2] - traj_ref[:, :2]) ** 2, axis=-1)))
            < thres
        )

    for i in range(gt_fut_traj.shape[0]):
        ts = ts_limit[i]
        x_curr = [bbox_tensor[i, 0], bbox_tensor[i, 1], yaw_preds[i], speed_preds[i]]
        reference_trajectory = np.concatenate(
            [gt_fut_traj[i], gt_fut_traj_yaw[i]], axis=-1
        )
        if (
            ts > 1
            and _is_dynamic(gt_fut_traj[i], int(ts), 2)
            and _check_diff(x_curr, reference_trajectory)
        ):
            smoother = MotionNonlinearSmoother(trajectory_len=int(ts), dt=0.5)
            reference_trajectory = reference_trajectory[: int(ts) + 1, :]
            smoother.set_reference_trajectory(x_curr, reference_trajectory)
            sol = smoother.solve()
            traj_perturb = np.stack(
                [sol.value(smoother.position_x), sol.value(smoother.position_y)],
                axis=-1,
            )
            if not _check_ade(traj_perturb, reference_trajectory, thres=1.5):
                traj_perturb = gt_fut_traj[i, 1:, :2] - gt_fut_traj[i, 0:1, :2]
            else:
                traj_perturb_tmp = traj_perturb[1:, :2] - traj_perturb[0:1, :2]
                traj_perturb = np.zeros((12, 2))
                traj_perturb[: traj_perturb_tmp.shape[0], :] = traj_perturb_tmp[:, :2]
        else:
            traj_perturb = gt_fut_traj[i, 1:, :2] - gt_fut_traj[i, 0:1, :2]
        traj_perturb_all.append(traj_perturb)
    return np.array(traj_perturb_all), (gt_fut_traj_mask > 0)
