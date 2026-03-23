# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: seg_head_plugin/seg_detr_head.py — Pure Python/numpy replacement.
SegDETRHead is a training-only component; replaced with a stub.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from ....ttsim_utils import (  # type: ignore[import-not-found]
    HEADS,
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
    build_loss,
)


@HEADS.register_module()
class SegDETRHead(SimNN.Module):
    """
    TTSim stub for SegDETRHead.
    Full training implementation replaced with simulation-compatible stub.
    """

    _version = 2

    def __init__(
        self,
        num_classes,
        num_things_classes,
        num_stuff_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        transformer=None,
        sync_cls_avg_factor=False,
        positional_encoding=None,
        loss_cls=None,
        loss_bbox=None,
        loss_iou=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        embed_dims=256,
        **kwargs,
    ):
        super().__init__()
        self.name = "seg_detr_head"
        self.num_classes = num_classes
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.in_channels = in_channels
        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.embed_dims = embed_dims
        self.test_cfg = test_cfg or {}

        cls_out_channels = num_things_classes + 1

        # SimNN layers
        self.input_proj = F.Conv2d(
            self.name + ".input_proj", in_channels, embed_dims, kernel_size=1
        )
        self.fc_cls = SimNN.Linear(self.name + ".fc_cls", embed_dims, cls_out_channels)
        self.fc_reg = SimNN.Linear(self.name + ".fc_reg", embed_dims, 4)

        self.relu = F.Relu(self.name + ".relu")

    def __call__(self, feats, img_metas=None):
        raise NotImplementedError(
            "SegDETRHead.__call__: use ttsim simulation path from seg_head.py"
        )

    def forward(self, feats, img_metas=None):
        return self(feats, img_metas)

    def loss(self, *args, **kwargs):
        raise NotImplementedError("SegDETRHead.loss: training-only")

    def get_bboxes(self, *args, **kwargs):
        raise NotImplementedError("SegDETRHead.get_bboxes: use ttsim simulation path")
