# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: seg_head_plugin/seg_assigner.py — Pure Python/numpy replacement.
Training-only classes (assigners, samplers) are stubs that raise NotImplementedError.
"""

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]

from ....ttsim_utils import BBOX_ASSIGNERS, BBOX_SAMPLERS, bbox_cxcywh_to_xyxy  # type: ignore[import-not-found]

INF = 10000000


class AssignResult:
    """Minimal stub replacing mmdet AssignResult."""

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels


class SamplingResult_segformer:
    """Pure Python sampling result for segformer."""

    def __init__(
        self, pos_inds, neg_inds, bboxes, gt_bboxes, gt_masks, assign_result, gt_flags
    ):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.size == 0:
            assert self.pos_assigned_gt_inds.size == 0
            self.pos_gt_bboxes = gt_bboxes.reshape(-1, 4)
            n, h, w = gt_masks.shape
            self.pos_gt_masks = gt_masks.reshape(-1, h, w)
        else:
            if gt_bboxes.ndim < 2:
                gt_bboxes = gt_bboxes.reshape(-1, 4)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
            self.pos_gt_masks = gt_masks[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return np.concatenate([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        return self


@BBOX_SAMPLERS.register_module()
class PseudoSampler_segformer:
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, gt_masks, **kwargs):
        bboxes = np.asarray(bboxes)
        gt_bboxes = np.asarray(gt_bboxes)
        gt_inds = np.asarray(assign_result.gt_inds)
        pos_inds = np.where(gt_inds > 0)[0]
        neg_inds = np.where(gt_inds == 0)[0]
        gt_flags = np.zeros(bboxes.shape[0], dtype=np.uint8)
        return SamplingResult_segformer(
            pos_inds, neg_inds, bboxes, gt_bboxes, gt_masks, assign_result, gt_flags
        )


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner_filter:
    """Hungarian assigner with filter support — training-only stub."""

    def __init__(self, cls_cost=None, reg_cost=None, iou_cost=None, max_pos=3):
        self.max_pos = max_pos

    def assign(
        self,
        bbox_pred,
        cls_pred,
        gt_bboxes,
        gt_labels,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        raise NotImplementedError("HungarianAssigner_filter.assign: training-only")


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner_multi_info:
    """Hungarian assigner with multi-info support — training-only stub."""

    def __init__(self, cls_cost=None, reg_cost=None, iou_cost=None, mask_cost=None):
        pass

    def assign(
        self,
        bbox_pred,
        cls_pred,
        mask_pred,
        gt_bboxes,
        gt_labels,
        gt_mask,
        img_meta,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        raise NotImplementedError("HungarianAssigner_multi_info.assign: training-only")
