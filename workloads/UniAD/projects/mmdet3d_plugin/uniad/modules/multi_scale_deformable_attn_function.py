# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
TTSim: multi_scale_deformable_attn_function.py
CUDA extension replaced by ttsim simulation ops.
multi_scale_deformable_attn_ttsim implemented inline (no reference/ imports).
"""

import numpy as np
import ttsim.front.functional.op as F


def multi_scale_deformable_attn_ttsim(
    name,
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    debug=False,
):
    """
    TTSim implementation of multi-scale deformable attention.

    Args:
        name (str): Operation name prefix
        value: SimTensor [bs, num_keys, num_heads, embed_dims_per_head]
        value_spatial_shapes: list of (H, W) tuples or SimTensor [num_levels, 2]
        sampling_locations: SimTensor [bs, num_queries, num_heads, num_levels, num_points, 2]
        attention_weights: SimTensor [bs, num_queries, num_heads, num_levels, num_points]

    Returns:
        SimTensor [bs, num_queries, embed_dims]
    """
    bs = value.shape[0]
    num_heads = value.shape[2]
    embed_dims_per_head = value.shape[3]

    num_queries = sampling_locations.shape[1]
    num_levels = sampling_locations.shape[3]
    num_points = sampling_locations.shape[4]

    if isinstance(value_spatial_shapes, list):
        spatial_shapes_list = value_spatial_shapes
    else:
        spatial_shapes_list = [
            (int(value_spatial_shapes.data[i, 0]), int(value_spatial_shapes.data[i, 1]))
            for i in range(num_levels)
        ]

    def _check(tensor, label):
        if debug and tensor.data is None:
            raise RuntimeError(f"{label} has no data")

    # ── Step 1: split value by spatial level ─────────────────────────────────
    split_sizes = [H * W for H, W in spatial_shapes_list]
    value_list = []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        v_level = F.SliceF(
            name + f".value_split_{start_idx}",
            out_shape=[bs, end_idx - start_idx, num_heads, embed_dims_per_head],
        )(
            value,
            F._from_data(
                name + f".value_split_{start_idx}.starts",
                np.array([start_idx], dtype=np.int64),
                is_const=True,
            ),
            F._from_data(
                name + f".value_split_{start_idx}.ends",
                np.array([end_idx], dtype=np.int64),
                is_const=True,
            ),
            F._from_data(
                name + f".value_split_{start_idx}.axes",
                np.array([1], dtype=np.int64),
                is_const=True,
            ),
            F._from_data(
                name + f".value_split_{start_idx}.steps",
                np.array([1], dtype=np.int64),
                is_const=True,
            ),
        )
        _check(v_level, f"value_level_{start_idx}")
        value_list.append(v_level)
        start_idx = end_idx

    # ── Step 2: normalize sampling locations to [-1, 1] ──────────────────────
    two = F._from_data(name + ".two", np.array(2.0, dtype=np.float32), is_const=True)
    one = F._from_data(name + ".one", np.array(1.0, dtype=np.float32), is_const=True)
    sampling_grids = F.Sub(name + ".sampling_sub1")(
        F.Mul(name + ".sampling_mul2")(sampling_locations, two), one
    )

    # ── Step 3: per-level grid sample ─────────────────────────────────────────
    sampling_value_list = []
    for level, (H, W) in enumerate(spatial_shapes_list):
        v_l = value_list[level]

        # [bs, H*W, num_heads, dH] -> [bs, H*W, num_heads*dH]
        v_l_flat = F.Reshape(name + f".value_l{level}_flat1")(
            v_l,
            F._from_data(
                name + f".value_l{level}_shape1",
                np.array([bs, H * W, num_heads * embed_dims_per_head], dtype=np.int64),
                is_const=True,
            ),
        )
        # -> [bs, num_heads*dH, H*W]
        v_l_trans = F.Transpose(name + f".value_l{level}_trans", perm=[0, 2, 1])(
            v_l_flat
        )
        # -> [bs*num_heads, dH, H, W]
        v_l_img = F.Reshape(name + f".value_l{level}_img")(
            v_l_trans,
            F._from_data(
                name + f".value_l{level}_img_shape",
                np.array([bs * num_heads, embed_dims_per_head, H, W], dtype=np.int64),
                is_const=True,
            ),
        )

        # extract level slice from sampling_grids: [bs, nQ, nH, 1, nP, 2]
        grid_l = F.SliceF(
            name + f".sampling_grid_l{level}",
            out_shape=[bs, num_queries, num_heads, 1, num_points, 2],
        )(
            sampling_grids,
            F._from_data(
                name + f".sampling_grid_l{level}.starts",
                np.array([level], dtype=np.int64),
                is_const=True,
            ),
            F._from_data(
                name + f".sampling_grid_l{level}.ends",
                np.array([level + 1], dtype=np.int64),
                is_const=True,
            ),
            F._from_data(
                name + f".sampling_grid_l{level}.axes",
                np.array([3], dtype=np.int64),
                is_const=True,
            ),
            F._from_data(
                name + f".sampling_grid_l{level}.steps",
                np.array([1], dtype=np.int64),
                is_const=True,
            ),
        )
        # squeeze level dim: [bs, nQ, nH, nP, 2]
        grid_l = F.Squeeze(name + f".sampling_grid_l{level}_sq")(
            grid_l,
            F._from_data(
                name + f".sampling_grid_l{level}_sq.axes",
                np.array([3], dtype=np.int64),
                is_const=True,
            ),
        )
        # [bs, nH, nQ, nP, 2]
        grid_l = F.Transpose(
            name + f".sampling_grid_l{level}_trans", perm=[0, 2, 1, 3, 4]
        )(grid_l)
        # [bs*nH, nQ, nP, 2]
        grid_l_flat = F.Reshape(name + f".sampling_grid_l{level}_flat")(
            grid_l,
            F._from_data(
                name + f".sampling_grid_l{level}_flat_shape",
                np.array([bs * num_heads, num_queries, num_points, 2], dtype=np.int64),
                is_const=True,
            ),
        )

        # grid_sample -> [bs*nH, dH, nQ, nP]
        sv_l = F.GridSample(
            name + f".grid_sample_l{level}",
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )(v_l_img, grid_l_flat)
        _check(sv_l, f"sampling_value_l{level}")
        sampling_value_list.append(sv_l)

    # ── Step 4: stack + weighted sum ─────────────────────────────────────────
    unsq_vals = []
    for level, sv_l in enumerate(sampling_value_list):
        sv_l_unsq = F.Unsqueeze(name + f".sampling_value_l{level}.unsq")(
            sv_l,
            F._from_data(
                name + f".sampling_value_l{level}.unsq_axes",
                np.array([3], dtype=np.int64),
                is_const=True,
            ),
        )
        unsq_vals.append(sv_l_unsq)

    stacked = (
        unsq_vals[0]
        if len(unsq_vals) == 1
        else F.ConcatX(name + ".concat_levels", axis=3)(*unsq_vals)
    )

    # [bs*nH, dH, nQ, nL*nP]
    stacked_flat = F.Reshape(name + ".stacked_flat")(
        stacked,
        F._from_data(
            name + ".stacked_flat_shape",
            np.array(
                [
                    bs * num_heads,
                    embed_dims_per_head,
                    num_queries,
                    num_levels * num_points,
                ],
                dtype=np.int64,
            ),
            is_const=True,
        ),
    )

    # attention_weights: [bs, nQ, nH, nL, nP] -> [bs*nH, 1, nQ, nL*nP]
    attn_flat = F.Reshape(name + ".attn_flat")(
        F.Transpose(name + ".attn_trans", perm=[0, 2, 1, 3, 4])(attention_weights),
        F._from_data(
            name + ".attn_flat_shape",
            np.array(
                [bs * num_heads, 1, num_queries, num_levels * num_points],
                dtype=np.int64,
            ),
            is_const=True,
        ),
    )

    weighted = F.Mul(name + ".weighted")(stacked_flat, attn_flat)
    aggregated = F.ReduceSum(name + ".aggregate", axis=3, keepdims=False)(weighted)

    # [bs, nH*dH, nQ]
    output = F.Reshape(name + ".output_reshape")(
        aggregated,
        F._from_data(
            name + ".output_shape",
            np.array(
                [bs, num_heads * embed_dims_per_head, num_queries], dtype=np.int64
            ),
            is_const=True,
        ),
    )
    # [bs, nQ, embed_dims]
    output = F.Transpose(name + ".output_final", perm=[0, 2, 1])(output)
    return output


class MultiScaleDeformableAttnFunction_fp32:
    """TTSim: delegates to multi_scale_deformable_attn_ttsim (inline above)."""

    @staticmethod
    def apply(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        return multi_scale_deformable_attn_ttsim(
            "ms_deform_attn",
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
        )


# fp16 is identical to fp32 in simulation (no precision distinction)
MultiScaleDeformableAttnFunction_fp16 = MultiScaleDeformableAttnFunction_fp32
