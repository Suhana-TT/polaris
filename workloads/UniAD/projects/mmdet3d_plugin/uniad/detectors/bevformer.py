# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
TTSim: detectors/bevformer.py — SimNN stub for BEVFormer.
No torch, no mmcv, no mmdet, no mmdet3d imports.
"""

import copy
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from ..ttsim_utils import DETECTORS, bbox3d2result, auto_fp16


@DETECTORS.register_module()
class BEVFormer(SimNN.Module):
    """
    TTSim stub for BEVFormer detector.
    Full training/testing implementation replaced with simulation-compatible stub.

    In simulation, the pts_bbox_head submodule drives the BEV feature extraction
    and detection pipeline. This class provides the structural wrapper with
    prev_frame_info state tracking.
    """

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):
        super().__init__()
        self.name = "bevformer"
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode

        # Temporal state (pure Python, not SimNN)
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        # pts_bbox_head is the main SimNN submodule; built externally and assigned
        # by the model builder (ttsim_utils.build_transformer_layer_sequence etc.)
        self.pts_bbox_head = None  # set by subclass or builder

        super().link_op2module()

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Stub: image feature extraction not simulated at this level."""
        raise NotImplementedError(
            "BEVFormer.extract_img_feat: use pts_bbox_head directly"
        )

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Stub: feature extraction not simulated at this level."""
        raise NotImplementedError("BEVFormer.extract_feat: use pts_bbox_head directly")

    def forward_pts_train(
        self,
        pts_feats,
        gt_bboxes_3d,
        gt_labels_3d,
        img_metas,
        gt_bboxes_ignore=None,
        prev_bev=None,
    ):
        """Training-only: raises NotImplementedError."""
        raise NotImplementedError("BEVFormer.forward_pts_train: training-only")

    def forward_train(self, **kwargs):
        """Training-only: raises NotImplementedError."""
        raise NotImplementedError("BEVFormer.forward_train: training-only")

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Stub: history BEV not simulated."""
        raise NotImplementedError("BEVFormer.obtain_history_bev: not simulated")

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Run pts_bbox_head and return (bev_embed, bbox_results)."""
        if self.pts_bbox_head is None:
            raise RuntimeError("BEVFormer: pts_bbox_head not set")
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentation."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list: list[dict] = [dict() for _ in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list

    def forward_test(self, img_metas, img=None, **kwargs):
        """Inference forward with temporal state tracking."""
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be a list, got {type(img_metas)}")
        img = [img] if img is None else img

        scene_token = img_metas[0][0].get("scene_token", None)
        if scene_token != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = scene_token

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Update can_bus delta
        can_bus = img_metas[0][0].get("can_bus", None)
        if can_bus is not None:
            tmp_pos = copy.deepcopy(np.asarray(can_bus[:3]))
            tmp_angle = copy.deepcopy(float(can_bus[-1]))
            if self.prev_frame_info["prev_bev"] is not None:
                img_metas[0][0]["can_bus"][:3] = (
                    np.asarray(can_bus[:3]) - self.prev_frame_info["prev_pos"]
                )
                img_metas[0][0]["can_bus"][-1] = (
                    float(can_bus[-1]) - self.prev_frame_info["prev_angle"]  # type: ignore[operator]
                )
            else:
                img_metas[0][0]["can_bus"][-1] = 0
                img_metas[0][0]["can_bus"][:3] = 0
            self.prev_frame_info["prev_pos"] = tmp_pos  # type: ignore[assignment]
            self.prev_frame_info["prev_angle"] = tmp_angle  # type: ignore[assignment]

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def __call__(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
