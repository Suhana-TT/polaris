# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

"""
TTSim: modules/decoder.py
- DetectionTransformerDecoder: stub (full impl in track_head.py)
- CustomMSDeformableAttention: full SimNN module using multi_scale_deformable_attn_ttsim
- inverse_sigmoid: pure numpy
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from .multi_scale_deformable_attn_function import multi_scale_deformable_attn_ttsim
from ..ttsim_utils import ATTENTION, xavier_init, constant_init


def inverse_sigmoid(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.clip(x, eps, 1 - eps) / np.clip(1 - x, eps, 1 - eps))


class DetectionTransformerDecoder(SimNN.Module):
    """TTSim stub: full decoder implementation in track_head.py's _DecoderLayer."""

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super().__init__()
        self.name = "detection_transformer_decoder"
        self.return_intermediate = return_intermediate

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Use track_head.py decoder directly")


@ATTENTION.register_module()
class CustomMSDeformableAttention(SimNN.Module):
    """TTSim SimNN module for multi-scale deformable attention."""

    def __init__(
        self,
        name="cmsda",
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.batch_first = batch_first

        self.sampling_offsets = SimNN.Linear(
            name + ".offsets", embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = SimNN.Linear(
            name + ".weights", embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = SimNN.Linear(name + ".value_proj", embed_dims, embed_dims)
        self.output_proj = SimNN.Linear(name + ".output_proj", embed_dims, embed_dims)

        self.softmax = F.Softmax(name + ".softmax", axis=-1)
        self.dropout = F.Dropout(name + ".drop", dropout, True)
    
    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs
    ):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = F.Add(self.name + ".q_pos_add")(query, query_pos)
    
        value_out = self.value_proj(value)
        offsets = self.sampling_offsets(query)
        weights = self.attention_weights(query)
    
        # 1. Get batch size and num_queries
        bs, num_queries = query.shape[:2]
    
        # 2. Reshape offsets to correct shape
        offsets = F.Reshape(self.name + ".reshape_offsets", (bs, num_queries, self.num_heads, self.num_levels, self.num_points, 2))(offsets)
    
        # 3. Reshape weights to correct shape
        wweights = F.Reshape(self.name + ".reshape_weights", (bs, num_queries, self.num_heads, self.num_levels, self.num_points))(weights)
        weights = self.softmax(weights)
    
        # 4. Compute sampling locations using offsets
        sampling_locations = F.Add(self.name + ".add_offsets")(reference_points, offsets)
    
        # Deformable attention core (ttsim)
        output = multi_scale_deformable_attn_ttsim(
            self.name + ".msda", value_out, spatial_shapes, sampling_locations, weights
        )
    
        output = self.output_proj(output)
        output = self.dropout(output)
        return F.Add(self.name + ".residual")(identity, output)
