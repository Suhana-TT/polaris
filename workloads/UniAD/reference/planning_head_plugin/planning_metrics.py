# SPDX-FileCopyrightText: (C) 2020 SenseTime. All Rights Reserved
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#

"""
TTSim: planning_head_plugin/planning_metrics.py — Pure Python/numpy replacement.
pytorch_lightning.Metric base class replaced with plain Python class.
"""

import numpy as np
from skimage.draw import polygon  # type: ignore[import-not-found]
from ..occ_head_plugin import calculate_birds_eye_view_parameters, gen_dx_bx


class _Metric:
    """Plain Python base class replacing pytorch_lightning.metrics.Metric."""

    def __init__(self, compute_on_step: bool = False):
        self._states: dict = {}

    def add_state(self, name: str, default, dist_reduce_fx=None):
        arr = (
            np.array(default) if not isinstance(default, np.ndarray) else default.copy()
        )
        self._states[name] = arr
        setattr(self, name, arr)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        for name, default in self._states.items():
            setattr(self, name, default.copy())


class PlanningMetric(_Metric):
    obj_col: np.ndarray
    obj_box_col: np.ndarray
    L2: np.ndarray
    total: np.ndarray

    def __init__(self, n_future=6, compute_on_step: bool = False):
        super().__init__(compute_on_step=compute_on_step)
        dx, bx, _ = gen_dx_bx(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )
        dx, bx = dx[:2], bx[:2]
        self.dx = dx  # numpy array
        self.bx = bx  # numpy array

        _, _, self.bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.W = 1.85
        self.H = 4.084
        self.n_future = n_future

        self.add_state("obj_col", default=np.zeros(self.n_future))
        self.add_state("obj_box_col", default=np.zeros(self.n_future))
        self.add_state("L2", default=np.zeros(self.n_future))
        self.add_state("total", default=np.array(0))

    def evaluate_single_coll(self, traj, segmentation):
        """
        traj: np.ndarray (n_future, 2)
        segmentation: np.ndarray (n_future, H, W)
        """
        pts = np.array(
            [
                [-self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, -self.W / 2.0],
                [-self.H / 2.0 + 0.5, -self.W / 2.0],
            ]
        )
        pts = (pts - self.bx) / self.dx
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.reshape(n_future, 1, 2)
        trajs = trajs[:, :, [1, 0]]  # swap x,y
        trajs = trajs / self.dx
        trajs = trajs + rc  # (n_future, 32, 2)

        r = trajs[:, :, 0].astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)
        c = trajs[:, :, 1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr_ = r[t]
            cc_ = c[t]
            I = np.logical_and(
                np.logical_and(rr_ >= 0, rr_ < self.bev_dimension[0]),
                np.logical_and(cc_ >= 0, cc_ < self.bev_dimension[1]),
            )
            seg_t = np.asarray(segmentation[t])
            collision[t] = np.any(seg_t[rr_[I], cc_[I]])
        return collision

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        """
        trajs: np.ndarray (B, n_future, 2)
        gt_trajs: np.ndarray (B, n_future, 2)
        segmentation: np.ndarray (B, n_future, H, W)
        """
        trajs = np.asarray(trajs)
        gt_trajs = np.asarray(gt_trajs)
        segmentation = np.asarray(segmentation)
        B, n_future, _ = trajs.shape
        trajs = trajs * np.array([-1, 1])
        gt_trajs = gt_trajs * np.array([-1, 1])

        obj_coll_sum = np.zeros(n_future)
        obj_box_coll_sum = np.zeros(n_future)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], segmentation[i])

            xx = trajs[i, :, 0]
            yy = trajs[i, :, 1]
            yi = ((yy - self.bx[0]) / self.dx[0]).astype(np.int64)
            xi = ((xx - self.bx[1]) / self.dx[1]).astype(np.int64)

            m1 = np.logical_and(
                np.logical_and(yi >= 0, yi < self.bev_dimension[0]),
                np.logical_and(xi >= 0, xi < self.bev_dimension[1]),
            )
            m1 = np.logical_and(m1, ~gt_box_coll)

            ti = np.arange(n_future)
            seg_arr = np.asarray(segmentation[i])
            obj_coll_sum[ti[m1]] += seg_arr[ti[m1], yi[m1], xi[m1]].astype(np.int64)

            m2 = ~gt_box_coll
            box_coll = self.evaluate_single_coll(trajs[i], segmentation[i])
            obj_box_coll_sum[ti[m2]] += box_coll[ti[m2]].astype(np.int64)

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs, gt_trajs_mask):
        trajs = np.asarray(trajs)
        gt_trajs = np.asarray(gt_trajs)
        gt_trajs_mask = np.asarray(gt_trajs_mask)
        return np.sqrt(
            (((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2) * gt_trajs_mask).sum(axis=-1)
        )

    def update(self, trajs, gt_trajs, gt_trajs_mask, segmentation):
        trajs = np.asarray(trajs)
        gt_trajs = np.asarray(gt_trajs)
        gt_trajs_mask = np.asarray(gt_trajs_mask)
        assert trajs.shape == gt_trajs.shape
        trajs[..., 0] = -trajs[..., 0]
        gt_trajs[..., 0] = -gt_trajs[..., 0]
        L2 = self.compute_L2(trajs, gt_trajs, gt_trajs_mask)
        obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(
            trajs[:, :, :2], gt_trajs[:, :, :2], segmentation
        )

        self.obj_col += obj_coll_sum
        self.obj_box_col += obj_box_coll_sum
        self.L2 += L2.sum(axis=0)
        self.total += len(trajs)

    def compute(self):
        return {
            "obj_col": self.obj_col / self.total,
            "obj_box_col": self.obj_box_col / self.total,
            "L2": self.L2 / self.total,
        }
