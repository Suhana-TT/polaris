# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: seg_head_plugin/seg_deformable_transformer.py — SimNN stub.
No torch, no mmcv, no einops imports.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from ....ttsim_utils import TRANSFORMER  # type: ignore[import-not-found]


@TRANSFORMER.register_module()
class SegDeformableTransformer(SimNN.Module):
    """
    TTSim stub for SegDeformableTransformer.
    Full training implementation replaced with simulation-compatible stub.
    """

    def __init__(
        self,
        as_two_stage=False,
        num_feature_levels=4,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        **kwargs
    ):
        super().__init__()
        self.name = "seg_deformable_transformer"
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        # embed_dims derived from encoder config if provided
        self.embed_dims = 256
        if encoder is not None and isinstance(encoder, dict):
            self.embed_dims = encoder.get("embed_dims", 256)

        # Init layers
        if not self.as_two_stage:
            self.reference_points = SimNN.Linear(
                self.name + ".reference_points", self.embed_dims, 2
            )
        else:
            self.enc_output = SimNN.Linear(
                self.name + ".enc_output", self.embed_dims, self.embed_dims
            )
            self.enc_output_norm = F.LayerNorm(
                self.name + ".enc_output_norm", self.embed_dims
            )
            self.pos_trans = SimNN.Linear(
                self.name + ".pos_trans", self.embed_dims * 2, self.embed_dims * 2
            )
            self.pos_trans_norm = F.LayerNorm(
                self.name + ".pos_trans_norm", self.embed_dims * 2
            )

        super().link_op2module()

    def __call__(
        self,
        mlvl_feats,
        mlvl_masks,
        query_embed,
        mlvl_pos_embeds,
        reg_branches=None,
        cls_branches=None,
        **kwargs
    ):
        raise NotImplementedError(
            "SegDeformableTransformer.__call__: use ttsim simulation path from seg_head.py"
        )

    def forward(self, *args, **kwargs):
        return self(*args, **kwargs)
