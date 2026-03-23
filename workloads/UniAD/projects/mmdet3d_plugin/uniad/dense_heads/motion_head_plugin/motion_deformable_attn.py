# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: motion_head_plugin/motion_deformable_attn.py — SimNN replacements.
No torch, no mmcv, no einops imports.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from ....ttsim_utils import ATTENTION, TRANSFORMER_LAYER  # type: ignore[import-not-found]
from ....modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32  # type: ignore[import-not-found]


@TRANSFORMER_LAYER.register_module()
class MotionTransformerAttentionLayer(SimNN.Module):
    """TTSim SimNN stub for MotionTransformerAttentionLayer."""

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=None,
        operation_order=None,
        norm_cfg=None,
        init_cfg=None,
        batch_first=False,
        **kwargs
    ):
        super().__init__()
        self.name = "motion_transformer_attn_layer"
        self.operation_order = operation_order or ("cross_attn", "norm", "ffn", "norm")
        self.batch_first = batch_first

        # Determine embed_dims from attn_cfgs
        embed_dims = 256
        if attn_cfgs is not None:
            if isinstance(attn_cfgs, dict):
                embed_dims = attn_cfgs.get("embed_dims", embed_dims)
            elif isinstance(attn_cfgs, (list, tuple)) and len(attn_cfgs) > 0:
                embed_dims = attn_cfgs[0].get("embed_dims", embed_dims)
        self.embed_dims = embed_dims

        # FFN
        ffn_hidden = 1024
        if ffn_cfgs is not None and isinstance(ffn_cfgs, dict):
            ffn_hidden = ffn_cfgs.get("feedforward_channels", ffn_hidden)

        self.ffn_l1 = SimNN.Linear(self.name + ".ffn.l1", embed_dims, ffn_hidden)
        self.ffn_relu = F.Relu(self.name + ".ffn.relu")
        self.ffn_l2 = SimNN.Linear(self.name + ".ffn.l2", ffn_hidden, embed_dims)
        self.ffn_drop = F.Dropout(self.name + ".ffn.drop", 0.0, True)

        # Norms
        self.norm1 = F.LayerNorm(self.name + ".norm1", embed_dims)
        self.norm2 = F.LayerNorm(self.name + ".norm2", embed_dims)

        self.add1 = F.Add(self.name + ".add1")
        self.add2 = F.Add(self.name + ".add2")

        super().link_op2module()

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs
    ):
        raise NotImplementedError(
            "MotionTransformerAttentionLayer.__call__: use ttsim simulation path from motion_head.py"
        )


@ATTENTION.register_module()
class MotionDeformableAttention(SimNN.Module):
    """TTSim SimNN MotionDeformableAttention."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_steps=1,
        sample_index=-1,
        im2col_step=64,
        dropout=0.1,
        bev_range=None,
        voxel_size=None,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.name = "motion_deformable_attn"
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_steps = num_steps
        self.sample_index = sample_index
        self.im2col_step = im2col_step
        self.bev_range = bev_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.batch_first = batch_first

        self.sampling_offsets = SimNN.Linear(
            self.name + ".sampling_offsets",
            embed_dims,
            num_heads * num_steps * num_levels * num_points * 2,
        )
        self.attention_weights = SimNN.Linear(
            self.name + ".attention_weights",
            embed_dims,
            num_heads * num_steps * num_levels * num_points,
        )
        self.value_proj = SimNN.Linear(
            self.name + ".value_proj", embed_dims, embed_dims
        )

        # output_proj: Linear -> LayerNorm -> ReLU
        self.output_proj_linear = SimNN.Linear(
            self.name + ".output_proj.linear", num_steps * embed_dims, embed_dims
        )
        self.output_proj_norm = F.LayerNorm(self.name + ".output_proj.norm", embed_dims)
        self.output_proj_relu = F.Relu(self.name + ".output_proj.relu")

        self.softmax = F.Softmax(self.name + ".softmax", axis=-1)
        self.dropout = F.Dropout(self.name + ".dropout", dropout, True)
        self.add = F.Add(self.name + ".add")

        super().link_op2module()

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        spatial_shapes=None,
        level_start_index=None,
        bbox_results=None,
        reference_trajs=None,
        flag="decoder",
        **kwargs
    ):
        # value projection
        v = self.value_proj(value if value is not None else query)
        # sampling offsets and attention weights
        offsets = self.sampling_offsets(query)
        weights = self.softmax(self.attention_weights(query))
        # output projection
        out = self.output_proj_relu(self.output_proj_norm(self.output_proj_linear(v)))
        out = self.dropout(out)
        id_val = identity if identity is not None else query
        return self.add(out, id_val)


@ATTENTION.register_module()
class CustomModeMultiheadAttention(SimNN.Module):
    """TTSim SimNN CustomModeMultiheadAttention."""

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=None,
        init_cfg=None,
        **kwargs
    ):
        super().__init__()
        self.name = "custom_mode_mha"
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        self.wq = SimNN.Linear(self.name + ".wq", embed_dims, embed_dims)
        self.wk = SimNN.Linear(self.name + ".wk", embed_dims, embed_dims)
        self.wv = SimNN.Linear(self.name + ".wv", embed_dims, embed_dims)
        self.wo = SimNN.Linear(self.name + ".wo", embed_dims, embed_dims)
        self.softmax = F.Softmax(self.name + ".softmax", axis=-1)
        self.attn_drop = F.Dropout(self.name + ".attn_drop", attn_drop, True)
        self.proj_drop = F.Dropout(self.name + ".proj_drop", proj_drop, True)

        drop_prob = 0.0
        if dropout_layer is not None and isinstance(dropout_layer, dict):
            drop_prob = dropout_layer.get("drop_prob", 0.0)
        self.dropout_layer = F.Dropout(self.name + ".dropout_layer", drop_prob, True)

        self.add = F.Add(self.name + ".add")

        super().link_op2module()

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs
    ):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        attn = self.attn_drop(self.softmax(q))
        out = self.proj_drop(self.wo(attn))
        out = self.dropout_layer(out)
        return self.add(identity, out)
