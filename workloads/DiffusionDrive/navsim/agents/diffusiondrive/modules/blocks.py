# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# TTSIM Version of GridSampleCrossBEVAttention
# (Torch reference code: reference/torch_code/modules/blocks.py)
# ============================================================================

import numpy as np

try:
    import ttsim.front.functional.op as F
    from ttsim.front.functional.sim_nn import Module as SimNN_Module
except ImportError as exc:
    raise ImportError(
        "The 'ttsim' package must be installed or otherwise made available on "
        "PYTHONPATH before importing "
        "'workloads.DiffusionDrive.navsim.agents.diffusiondrive.modules.blocks'. "
        "Do not rely on this module to modify sys.path at import time."
    ) from exc

class GridSampleCrossBEVAttention_TTSIM(SimNN_Module):
    """TTSIM implementation of GridSampleCrossBEVAttention."""

    def __init__(
        self,
        embed_dims,
        num_heads,
        num_levels=1,
        in_bev_dims=64,
        num_points=8,
        config=None,
        name_prefix=None,
    ):
        super(GridSampleCrossBEVAttention_TTSIM, self).__init__()
        p = name_prefix or "GridSampleCrossBEVAttention_TTSIM"
        self.name = p
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config

        # Attention weights linear projection
        self.attention_weights_linear = F.Linear(
            f"{p}_attn_w_lin", embed_dims, num_points
        )
        self.attention_weights_bias = F.Bias(f"{p}_attn_w_bias", [num_points])

        # Output projection
        self.output_proj_linear = F.Linear(f"{p}_out_proj_lin", embed_dims, embed_dims)
        self.output_proj_bias = F.Bias(f"{p}_out_proj_bias", [embed_dims])

        # Dropout
        self.dropout_op = F.Dropout(f"{p}_dropout", 0.1, False)

        # Value projection: Conv2d + Bias + ReLU
        self.value_proj_conv = F.Conv2d(
            f"{p}_val_proj_conv", in_bev_dims, 256, kernel_size=3, stride=1, padding=1
        )
        self.value_proj_bias = F.Bias(f"{p}_val_proj_bias", [1, 256, 1, 1])
        self.value_proj_relu = F.Relu(f"{p}_val_proj_relu")

        # Softmax for attention weights
        self.softmax = F.Softmax(f"{p}_softmax", axis=-1)

        # Grid sample operation
        self.grid_sample = F.GridSample(
            f"{p}_grid_sample",
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Split traj_points along last axis into 2 halves (y, x)
        self.split_traj = F.Split(f"{p}_split_traj", axis=3, count=2)

        # Normalize ops
        self.div_y = F.Div(f"{p}_norm_div_y")
        self.div_x = F.Div(f"{p}_norm_div_x")

        # Pre-create normalization constants once
        self._norm_y = F._from_data(
            f"{p}_norm_y",
            np.float32(self.config.lidar_max_y),
            is_const=True,
        )
        self._tensors[self._norm_y.name] = self._norm_y

        self._norm_x = F._from_data(
            f"{p}_norm_x",
            np.float32(self.config.lidar_max_x),
            is_const=True,
        )
        self._tensors[self._norm_x.name] = self._norm_x

        # Concat normalized [x, y] for grid
        self.concat_grid = F.ConcatX(f"{p}_concat_grid", axis=-1)

        # Reshape for attention weights
        self.reshape_attn = F.Reshape(f"{p}_reshape_attn")

        # Reshape for attention weights expansion
        self.reshape_attn_exp = F.Reshape(f"{p}_reshape_attn_exp")

        # Weighted multiply, reduce, permute, residual add
        self.mul_weighted = F.Mul(f"{p}_mul_weighted")
        self.reduce_sum = F.ReduceSum(f"{p}_reduce_sum", axis=-1, keepdims=False)
        self.permute_out = F.permute(f"{p}_permute_out", [0, 2, 1])
        self.add_residual = F.Add(f"{p}_add_residual")

        # Link operations to module
        super().link_op2module()

    def init_weight(self):
        """Weight initialization is handled during weight injection."""
        pass

    def __call__(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Forward pass for BEV attention.

        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, in_bev_dims, height, width)
            spatial_shape: (height, width)

        Returns:
            output: attended features (bs, num_queries, embed_dims)
        """
        self._call_count = getattr(self, "_call_count", 0) + 1
        _cc = self._call_count

        bs = queries.shape[0]
        num_queries = queries.shape[1]
        num_points = traj_points.shape[2]

        # Split traj_points along last dim: (bs, nq, np, 2) → two (bs, nq, np, 1)
        traj_y, traj_x = self.split_traj(traj_points)

        # Reuse pre-created normalization constants
        norm_y = self._norm_y
        norm_x = self._norm_x

        normalized_y = self.div_y(traj_y, norm_y)
        normalized_x = self.div_x(traj_x, norm_x)

        # Swap x and y and concatenate: grid expects (x, y) but we have (y, x)
        normalized_trajectory = self.concat_grid(
            normalized_x, normalized_y
        )  # (bs, nq, np, 2)

        # Compute attention weights
        attention_weights = self.attention_weights_linear(queries)
        attention_weights = self.attention_weights_bias(attention_weights)

        # Reshape to (bs, num_queries, num_points)
        attn_shape = F._from_data(
            f"{self.name}_attn_shape_c{_cc}",
            np.array([bs, num_queries, num_points], dtype=np.int64),
            is_const=True,
        )
        self._tensors[attn_shape.name] = attn_shape
        attention_weights = self.reshape_attn(attention_weights, attn_shape)

        # Apply softmax
        attention_weights = self.softmax(attention_weights)

        # Project BEV features
        value = self.value_proj_conv(bev_feature)
        value = self.value_proj_bias(value)
        value = self.value_proj_relu(value)

        # Grid sample: (bs, C=256, H, W) x (bs, nq, np, 2) → (bs, C=256, nq, np)
        sampled_features = self.grid_sample(value, normalized_trajectory)

        # Expand attention_weights: (bs, nq, np) → (bs, 1, nq, np)
        attn_exp_shape = F._from_data(
            f"{self.name}_attn_exp_shape_c{_cc}",
            np.array([bs, 1, num_queries, num_points], dtype=np.int64),
            is_const=True,
        )
        self._tensors[attn_exp_shape.name] = attn_exp_shape
        attention_weights_expanded = self.reshape_attn_exp(
            attention_weights, attn_exp_shape
        )

        # Weighted sum: (bs, 1, nq, np) * (bs, C, nq, np) → sum over np → (bs, C, nq)
        weighted_features = self.mul_weighted(
            attention_weights_expanded, sampled_features
        )
        out = self.reduce_sum(weighted_features)  # (bs, C, num_queries)

        # Permute: (bs, C, nq) → (bs, nq, C)
        out = self.permute_out(out)

        # Output projection
        out = self.output_proj_linear(out)
        out = self.output_proj_bias(out)

        # Dropout + residual
        out = self.dropout_op(out)
        out = self.add_residual(out, queries)

        return out