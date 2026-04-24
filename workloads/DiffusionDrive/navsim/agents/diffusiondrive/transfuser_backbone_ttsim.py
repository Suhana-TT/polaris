#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSIM conversion of TransFuser vision backbone.
Converted from transfuser_backbone.py - preserves all logic, inputs, and outputs.

CONVERTED CLASSES:
------------------
1. GPT - Full GPT transformer backbone with positional embeddings
2. SelfAttention - Multi-head self-attention layer
3. Block - Transformer block with attention + MLP
4. MultiheadAttentionWithAttention - Attention layer that returns attention weights
5. TransformerDecoderLayerWithAttention - Decoder layer with self/cross attention
6. TransformerDecoderWithAttention - Full transformer decoder stack
7. TransfuserBackbone - Multi-scale fusion transformer for image + LiDAR

CONVERSION NOTES:
-----------------
- Custom layers (GPT, SelfAttention, Block, etc.) fully converted to TTSIM
- timm encoders (image_encoder, lidar_encoder) kept as PyTorch (external pretrained models)
- All nn.Module operations converted to ttsim equivalents (F.*, T.*)
- Preserves exact same forward pass logic and computational graph
- Linear layers split into F.Linear + F.Bias (PyTorch nn.Linear has bias, TTSIM F.Linear doesn't)
- Dropout uses positional args: F.Dropout(name, prob, train_mode, /, *, module=None)

KEY MAPPINGS:
- nn.Module → SimNN.Module
- nn.ModuleList → SimNN.ModuleList
- nn.Linear → F.Linear + F.Bias
- nn.Parameter → F._from_shape(..., is_param=True)
- torch.flatten(x, 1) → x.flatten(start_dim=1)
- F.interpolate(mode='bilinear') → F.Resize(mode='linear')
- torch.cat → F.ConcatX
- nn.Upsample → F.Resize (F.Upsample has issues with missing scales tensor)
- Arithmetic ops → F.Add, F.Multiply, etc.
"""

import os
import sys
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import copy
import math
import numpy as np

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor


class GPT(SimNN.Module):
    """GPT transformer backbone for TransFuser - TTSIM version."""

    def __init__(self, name, n_embd, config, lidar_time_frames):
        super().__init__()
        self.name = name
        self.n_embd = n_embd
        self.config = config
        self.lidar_time_frames = lidar_time_frames

        # Enforce current limitation: only seq_len = 1 is supported
        self.seq_len = getattr(config, "seq_len", 1)
        assert self.seq_len == 1, (
            f"this DiffusionDrive implementation only supports seq_len=1 "
            f"(got {self.seq_len})"
)
        self.lidar_seq_len = config.lidar_seq_len

        # Positional embedding parameter (learnable)
        pos_emb_size = (
            self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors
            + lidar_time_frames
            * self.config.lidar_vert_anchors
            * self.config.lidar_horz_anchors
        )
        self.pos_emb = F._from_shape(
            self.name + ".pos_emb",
            [1, pos_emb_size, self.n_embd],
            is_param=True,
            np_dtype=np.float32,
        )

        # Dropout (positional args: name, prob, train_mode)
        self.drop = F.Dropout(
            self.name + ".drop", config.embd_pdrop, False, module=self
        )

        # Transformer blocks
        blocks = []
        for layer in range(config.n_layer):
            blocks.append(
                Block(
                    f"{self.name}.blocks.{layer}",
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
            )
        self.blocks = SimNN.ModuleList(blocks)

        # Decoder head
        self.ln_f = F.LayerNorm(self.name + ".ln_f", n_embd)

        # Reusable operations
        self.add_pos_emb = F.Add(self.name + ".add_pos_emb")
        self.concat_tokens = F.ConcatX(self.name + ".concat_tokens", axis=1)
        self.tile_pos_emb = F.Tile(self.name + ".tile_pos_emb")

        super().link_op2module()

    def __call__(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor: [B*4*seq_len, C, H, W]
            lidar_tensor: [B*seq_len, C, H, W]
        """
        bz = lidar_tensor.shape[0]

        # Use ACTUAL input dimensions after avgpool, not config anchors
        # The avgpool in fuse_features resizes to anchor dimensions
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        # Reshape: (B, C, H, W) → (B, H, W, C) → (B, H*W, C)
        image_tensor = image_tensor.permute([0, 2, 3, 1])
        image_tensor = image_tensor.reshape([bz, -1, self.n_embd])

        lidar_tensor = lidar_tensor.permute([0, 2, 3, 1])
        lidar_tensor = lidar_tensor.reshape([bz, -1, self.n_embd])

        # Concatenate and add positional embedding
        token_embeddings = self.concat_tokens(image_tensor, lidar_tensor)

        # Dynamic positional embedding: resize to match actual sequence length
        actual_seq_len = token_embeddings.shape[1]
        assert self.pos_emb.shape is not None
        expected_seq_len = self.pos_emb.shape[1]

        if actual_seq_len != expected_seq_len:
            # Need to slice or repeat pos_emb to match actual_seq_len
            if actual_seq_len < expected_seq_len:
                # Slice pos_emb to match actual length
                # Use Python slicing: [:, :actual_seq_len, :]
                self.pos_emb.link_module = self  # Required for Python slicing
                pos_emb_adjusted = self.pos_emb[:, :actual_seq_len, :]  # type: ignore[index]
            else:
                # Tile/repeat pos_emb to match actual length
                # Repeat factor for axis 1
                repeat_factor = (
                    actual_seq_len + expected_seq_len - 1
                ) // expected_seq_len
                repeats = F._from_data(
                    self.name + ".tile_repeats",
                    np.array([1, repeat_factor, 1], dtype=np.int64),
                    is_const=True,
                )
                self._tensors[repeats.name] = repeats
                pos_emb_tiled = self.tile_pos_emb(self.pos_emb, repeats)
                pos_emb_tiled.link_module = self  # Required for Python slicing

                # Then slice to exact length using Python slicing
                pos_emb_adjusted = pos_emb_tiled[:, :actual_seq_len, :]
        else:
            pos_emb_adjusted = self.pos_emb

        x = self.add_pos_emb(token_embeddings, pos_emb_adjusted)
        x = self.drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Final layer norm
        x = self.ln_f(x)

        # Split back to image and lidar using Python slicing
        # Use ACTUAL spatial dimensions, not config
        img_spatial_len = img_h * img_w
        lidar_spatial_len = lidar_h * lidar_w

        # Link x to module for slicing
        x.link_module = self

        # Slice instead of using Split operation
        image_out = x[:, :img_spatial_len, :]
        lidar_out = x[:, img_spatial_len:, :]

        # Link output tensors to module for subsequent operations
        image_out.link_module = self
        lidar_out.link_module = self

        # Reshape back to spatial: [B, spatial_len, C] → [B, H, W, C] → [B, C, H, W]
        image_tensor_out = image_out.reshape(
            [bz * self.seq_len, img_h, img_w, self.n_embd]
        )
        image_tensor_out = image_tensor_out.permute([0, 3, 1, 2])

        lidar_tensor_out = lidar_out.reshape([bz, lidar_h, lidar_w, self.n_embd])
        lidar_tensor_out = lidar_tensor_out.permute([0, 3, 1, 2])

        return image_tensor_out, lidar_tensor_out


class SelfAttention(SimNN.Module):
    """Multi-head self-attention - TTSIM version."""

    def __init__(self, name, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.name = name
        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head

        # Q, K, V projections
        self.key = F.Linear(self.name + ".key", n_embd, n_embd, module=self)
        self.key_bias = F.Bias(self.name + ".key_bias", [n_embd])
        self.query = F.Linear(self.name + ".query", n_embd, n_embd, module=self)
        self.query_bias = F.Bias(self.name + ".query_bias", [n_embd])
        self.value = F.Linear(self.name + ".value", n_embd, n_embd, module=self)
        self.value_bias = F.Bias(self.name + ".value_bias", [n_embd])

        # Dropout (positional args: name, prob, train_mode)
        self.attn_drop = F.Dropout(
            self.name + ".attn_drop", attn_pdrop, False, module=self
        )
        self.resid_drop = F.Dropout(
            self.name + ".resid_drop", resid_pdrop, False, module=self
        )

        # Output projection
        self.proj = F.Linear(self.name + ".proj", n_embd, n_embd, module=self)
        self.proj_bias = F.Bias(self.name + ".proj_bias", [n_embd])

        # Scale factor
        self.scale = F._from_data(
            self.name + ".scale",
            np.float32(1.0 / math.sqrt(self.head_size)),
            is_const=True,
        )

        # Reusable ops
        self.scale_mul = F.Mul(self.name + ".scale_mul")
        self.softmax = F.Softmax(self.name + ".softmax", axis=-1)

        super().link_op2module()

    def __call__(self, x):
        b, t, c = x.shape

        # Compute Q, K, V and reshape for multi-head
        k = (
            self.key_bias(self.key(x))
            .reshape([b, t, self.n_head, self.head_size])
            .permute([0, 2, 1, 3])
        )
        q = (
            self.query_bias(self.query(x))
            .reshape([b, t, self.n_head, self.head_size])
            .permute([0, 2, 1, 3])
        )
        v = (
            self.value_bias(self.value(x))
            .reshape([b, t, self.n_head, self.head_size])
            .permute([0, 2, 1, 3])
        )

        # Attention: Q @ K^T * scale
        k_t = k.permute([0, 1, 3, 2])
        att = T.matmul(q, k_t)
        att = self.scale_mul(att, self.scale)
        att = self.softmax(att)
        att = self.attn_drop(att)

        # Apply attention to values
        y = T.matmul(att, v)

        # Reassemble heads
        y = y.permute([0, 2, 1, 3]).reshape([b, t, c])

        # Output projection
        y = self.proj_bias(self.proj(y))
        y = self.resid_drop(y)

        return y


class Block(SimNN.Module):
    """Transformer block - TTSIM version."""

    def __init__(self, name, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.name = name

        self.ln1 = F.LayerNorm(self.name + ".ln1", n_embd)
        self.ln2 = F.LayerNorm(self.name + ".ln2", n_embd)
        self.attn = SelfAttention(
            self.name + ".attn", n_embd, n_head, attn_pdrop, resid_pdrop
        )

        # MLP
        self.mlp_fc1 = F.Linear(
            self.name + ".mlp.0", n_embd, block_exp * n_embd, module=self
        )
        self.mlp_fc1_bias = F.Bias(self.name + ".mlp.0_bias", [block_exp * n_embd])
        self.mlp_relu = F.Relu(self.name + ".mlp.1")
        self.mlp_fc2 = F.Linear(
            self.name + ".mlp.2", block_exp * n_embd, n_embd, module=self
        )
        self.mlp_fc2_bias = F.Bias(self.name + ".mlp.2_bias", [n_embd])
        self.mlp_drop = F.Dropout(self.name + ".mlp.3", resid_pdrop, False, module=self)

        # Residual adds
        self.add1 = F.Add(self.name + ".add1")
        self.add2 = F.Add(self.name + ".add2")

        super().link_op2module()

    def __call__(self, x):
        # Attention with residual
        attn_out = self.attn(self.ln1(x))
        x = self.add1(x, attn_out)

        # MLP with residual
        mlp_out = self.mlp_fc1_bias(self.mlp_fc1(self.ln2(x)))
        mlp_out = self.mlp_relu(mlp_out)
        mlp_out = self.mlp_fc2_bias(self.mlp_fc2(mlp_out))
        mlp_out = self.mlp_drop(mlp_out)
        x = self.add2(x, mlp_out)

        return x


class MultiheadAttentionWithAttention(SimNN.Module):
    """
    MultiheadAttention that also returns attention weights - TTSIM version.
    """

    def __init__(self, name, n_embd, n_head, pdrop):
        super().__init__()
        self.name = name
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd

        # Key, query, value projections
        self.key = F.Linear(self.name + ".key", n_embd, n_embd, module=self)
        self.key_bias = F.Bias(self.name + ".key_bias", [n_embd])
        self.query = F.Linear(self.name + ".query", n_embd, n_embd, module=self)
        self.query_bias = F.Bias(self.name + ".query_bias", [n_embd])
        self.value = F.Linear(self.name + ".value", n_embd, n_embd, module=self)
        self.value_bias = F.Bias(self.name + ".value_bias", [n_embd])

        # Regularization
        self.attn_drop = F.Dropout(self.name + ".attn_drop", pdrop, False, module=self)
        self.resid_drop = F.Dropout(
            self.name + ".resid_drop", pdrop, False, module=self
        )

        # Output projection
        self.proj = F.Linear(self.name + ".proj", n_embd, n_embd, module=self)
        self.proj_bias = F.Bias(self.name + ".proj_bias", [n_embd])

        # Pre-allocate attention ops (must be in __init__ for get_ops() discovery)
        self.matmul_qk = F.MatMul(self.name + ".matmul_qk")
        self.mul_scale = F.Mul(self.name + ".mul_scale")
        self.softmax = F.Softmax(self.name + ".softmax", axis=-1)
        self.matmul_av = F.MatMul(self.name + ".matmul_av")
        self.reduce_mean_attention = F.ReduceMean(
            self.name + ".attention_mean", axes=1, keepdims=0
        )

        super().link_op2module()

    def __call__(self, q_in, k_in, v_in):
        b, t, c = q_in.shape[0], q_in.shape[1], q_in.shape[2]
        t_mem = k_in.shape[1]

        # Calculate query, key, values for all heads
        q = self.query_bias(self.query(q_in))  # (b, t, c)
        k = self.key_bias(self.key(k_in))  # (b, t_mem, c)
        v = self.value_bias(self.value(v_in))  # (b, t_mem, c)

        # Reshape for multi-head: (b, t, n_head, head_size)
        head_size = c // self.n_head
        q = q.reshape([b, t, self.n_head, head_size])
        k = k.reshape([b, t_mem, self.n_head, head_size])
        v = v.reshape([b, t_mem, self.n_head, head_size])

        # Transpose to (b, n_head, t, head_size)
        q = q.permute([0, 2, 1, 3])
        k = k.permute([0, 2, 1, 3])
        v = v.permute([0, 2, 1, 3])

        # Attention: (b, nh, t, hs) x (b, nh, hs, t_mem) -> (b, nh, t, t_mem)
        k_t = k.permute([0, 1, 3, 2])
        scale = F._from_data(
            self.name + ".scale",
            np.array(1.0 / math.sqrt(head_size), dtype=np.float32),
            is_const=True,
        )
        self._tensors[scale.name] = scale
        att = self.matmul_qk(q, k_t)
        att = self.mul_scale(att, scale)

        att = self.softmax(att)
        att = self.attn_drop(att)

        # Apply attention to values: (b, nh, t, t_mem) x (b, nh, t_mem, hs) -> (b, nh, t, hs)
        y = self.matmul_av(att, v)
        y.link_module = self  # Set link_module for subsequent permute operation

        # Transpose back: (b, nh, t, hs) -> (b, t, nh, hs)
        y = y.permute([0, 2, 1, 3])

        # Reshape: (b, t, nh, hs) -> (b, t, c)
        y = y.reshape([b, t, c])

        # Output projection
        y = self.resid_drop(self.proj_bias(self.proj(y)))

        # Average attention over heads: (b, nh, t, t_mem) -> (b, t, t_mem)
        attention = self.reduce_mean_attention(att)

        return y, attention


class TransformerDecoderLayerWithAttention(SimNN.Module):
    """A Transformer decoder layer that returns attentions - TTSIM version."""

    def __init__(
        self,
        name,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

        # Attention layers
        self.self_attn = MultiheadAttentionWithAttention(
            self.name + ".self_attn", d_model, nhead, dropout
        )
        self.multihead_attn = MultiheadAttentionWithAttention(
            self.name + ".multihead_attn", d_model, nhead, dropout
        )

        # Feedforward network
        self.linear1 = F.Linear(
            self.name + ".linear1", d_model, dim_feedforward, module=self
        )
        self.linear1_bias = F.Bias(self.name + ".linear1_bias", [dim_feedforward])
        self.dropout = F.Dropout(self.name + ".dropout", dropout, False, module=self)
        self.linear2 = F.Linear(
            self.name + ".linear2", dim_feedforward, d_model, module=self
        )
        self.linear2_bias = F.Bias(self.name + ".linear2_bias", [d_model])

        # Layer normalization
        self.norm1 = F.LayerNorm(self.name + ".norm1", d_model, eps=layer_norm_eps)
        self.norm2 = F.LayerNorm(self.name + ".norm2", d_model, eps=layer_norm_eps)
        self.norm3 = F.LayerNorm(self.name + ".norm3", d_model, eps=layer_norm_eps)

        # Dropout layers
        self.dropout1 = F.Dropout(self.name + ".dropout1", dropout, False, module=self)
        self.dropout2 = F.Dropout(self.name + ".dropout2", dropout, False, module=self)
        self.dropout3 = F.Dropout(self.name + ".dropout3", dropout, False, module=self)

        # Activation
        self.activation = F.Relu(self.name + ".activation")

        # Addition operations
        self.add1 = F.Add(self.name + ".add1")
        self.add2 = F.Add(self.name + ".add2")
        self.add3 = F.Add(self.name + ".add3")

        super().link_op2module()

    def __call__(self, tgt, memory):
        x = tgt

        # Self-attention with residual
        tmp, _ = self.self_attn(x, x, x)
        tmp = self.dropout1(tmp)
        x = self.norm1(self.add1(x, tmp))

        # Multi-head attention with residual
        tmp, attention = self.multihead_attn(x, memory, memory)
        tmp = self.dropout2(tmp)
        x = self.norm2(self.add2(x, tmp))

        # Feedforward with residual
        tmp = self.linear1_bias(self.linear1(x))
        tmp = self.activation(tmp)
        tmp = self.dropout(tmp)
        tmp = self.linear2_bias(self.linear2(tmp))
        tmp = self.dropout3(tmp)
        x = self.norm3(self.add3(x, tmp))

        return x, attention


class TransformerDecoderWithAttention(SimNN.Module):
    """A Transformer decoder that returns attentions - TTSIM version."""

    def __init__(self, name, layer_template, num_layers, norm=None):
        super().__init__()
        self.name = name
        self.num_layers = num_layers

        # Create layer copies
        layers = []
        for i in range(num_layers):
            # Create a new layer with copied parameters
            layer = TransformerDecoderLayerWithAttention(
                f"{self.name}.layers.{i}",
                layer_template.d_model,
                layer_template.self_attn.n_head,
                layer_template.dim_feedforward,
                # Extract dropout probability from layer template
                dropout=0.1,  # Default value, should be passed through
                layer_norm_eps=1e-5,
            )
            layers.append(layer)

        self.layers = SimNN.ModuleList(layers)

        # Optional final normalization
        if norm is not None:
            self.norm = F.LayerNorm(
                self.name + ".norm",
                layer_template.d_model,
                eps=1e-5,
            )
        else:
            self.norm = None

        # Pre-allocate reduce_mean for attention averaging
        self.avg_attention_mean = F.ReduceMean(
            self.name + ".avg_attention", axes=0, keepdims=0
        )

        super().link_op2module()

    def __call__(self, queries, memory):
        output = queries
        attentions = []

        for mod in self.layers:
            output, attention = mod(output, memory)
            # Ensure attention tensor has link_module for T.stack
            attention.link_module = self
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        # Stack attentions: list of (b, t, t_mem) -> (num_layers, b, t, t_mem)
        if len(attentions) > 0:
            stacked = T.stack(attentions, dim=0)
            # Average over layers: (num_layers, b, t, t_mem) -> (b, t, t_mem)
            avg_attention = self.avg_attention_mean(stacked)
        else:
            avg_attention = None

        return output, avg_attention


class BasicBlock_TTSIM(SimNN.Module):
    """ResNet BasicBlock fully in TTSIM."""

    expansion = 1

    def __init__(self, name, in_ch, out_ch, stride=1, downsample=False):
        super().__init__()
        self.name = name
        self.conv1 = F.Conv2d(
            f"{name}.conv1", in_ch, out_ch, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = F.BatchNorm2d(f"{name}.bn1", out_ch)
        self.relu1 = F.Relu(f"{name}.relu1")
        self.conv2 = F.Conv2d(
            f"{name}.conv2", out_ch, out_ch, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = F.BatchNorm2d(f"{name}.bn2", out_ch)
        self.add = F.Add(f"{name}.add")
        self.relu_out = F.Relu(f"{name}.relu_out")

        self.has_downsample = downsample
        if downsample:
            self.ds_conv = F.Conv2d(
                f"{name}.downsample.0", in_ch, out_ch, kernel_size=1, stride=stride
            )
            self.ds_bn = F.BatchNorm2d(f"{name}.downsample.1", out_ch)

        super().link_op2module()

    def __call__(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.has_downsample:
            identity = self.ds_bn(self.ds_conv(x))
        out = self.relu_out(self.add(out, identity))
        return out


class ResNetEncoder_TTSIM(SimNN.Module):
    """ResNet encoder fully in TTSIM (replaces timm pretrained model).

    Supports resnet18 [2,2,2,2] and resnet34 [3,4,6,3] with BasicBlock.
    Provides the same interface as timm features_only models:
      - return_layers, feature_info for TransfuserBackbone compatibility
      - forward_stem() / forward_stage() for direct stage-wise execution
    """

    ARCH_LAYERS = {
        "resnet18": [2, 2, 2, 2],
        "resnet34": [3, 4, 6, 3],
    }

    def __init__(self, name, in_channels=3, arch="resnet34"):
        super().__init__()
        self.name = name
        layers = self.ARCH_LAYERS.get(arch, [3, 4, 6, 3])
        channels = [64, 128, 256, 512]

        # Stem: conv1(7x7, s=2, p=3) + bn1 + relu
        self.conv1 = F.Conv2d(
            f"{name}.conv1", in_channels, 64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = F.BatchNorm2d(f"{name}.bn1", 64)
        self.act1 = F.Relu(f"{name}.act1")

        # MaxPool (3x3, s=2, p=1) — match timm's padding=1
        self.maxpool = F.MaxPool2d(
            f"{name}.maxpool", kernel_size=3, stride=2, padding=1
        )

        # 4 residual stages
        self.layer1 = self._make_layer(
            f"{name}.layer1", 64, channels[0], layers[0], stride=1
        )
        self.layer2 = self._make_layer(
            f"{name}.layer2", channels[0], channels[1], layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            f"{name}.layer3", channels[1], channels[2], layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            f"{name}.layer4", channels[2], channels[3], layers[3], stride=2
        )

        # Compatibility attributes used by TransfuserBackbone
        self.return_layers = {
            "act1": "0",
            "layer1": "1",
            "layer2": "2",
            "layer3": "3",
            "layer4": "4",
        }

        class _FeatureInfo:
            def __init__(self):
                self.info = [
                    {"num_chs": 64, "reduction": 2},  # act1
                    {"num_chs": 64, "reduction": 4},  # layer1
                    {"num_chs": 128, "reduction": 8},  # layer2
                    {"num_chs": 256, "reduction": 16},  # layer3
                    {"num_chs": 512, "reduction": 32},  # layer4
                ]

        self.feature_info = _FeatureInfo()

        super().link_op2module()

    def _make_layer(self, name, in_ch, out_ch, num_blocks, stride):
        blocks = []
        downsample = stride != 1 or in_ch != out_ch
        blocks.append(
            BasicBlock_TTSIM(
                f"{name}.0", in_ch, out_ch, stride=stride, downsample=downsample
            )
        )
        for i in range(1, num_blocks):
            blocks.append(BasicBlock_TTSIM(f"{name}.{i}", out_ch, out_ch))
        return SimNN.ModuleList(blocks)

    def forward_stem(self, x):
        """conv1 -> bn1 -> relu (returns after act1)."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

    def forward_stage(self, stage_idx, x):
        """Run one residual stage (0=maxpool+layer1, 1=layer2, 2=layer3, 3=layer4)."""
        if stage_idx == 0:
            x = self.maxpool(x)
            for blk in self.layer1:
                x = blk(x)
        elif stage_idx == 1:
            for blk in self.layer2:
                x = blk(x)
        elif stage_idx == 2:
            for blk in self.layer3:
                x = blk(x)
        elif stage_idx == 3:
            for blk in self.layer4:
                x = blk(x)
        return x


class TransfuserBackbone(SimNN.Module):
    """Multi-scale Fusion Transformer for image + LiDAR - TTSIM version."""

    def __init__(self, name, config):
        super().__init__()
        self.name = name
        self.config = config

        # Image encoder — fully TTSIM ResNet
        self.image_encoder = ResNetEncoder_TTSIM(
            f"{name}.image_encoder",
            in_channels=3,
            arch=config.image_architecture,
        )

        # Channel configuration
        if config.use_ground_plane:
            in_channels = 2 * config.lidar_seq_len
        else:
            in_channels = config.lidar_seq_len

        # Latent lidar (if used)
        if config.latent:
            self.lidar_latent = F._from_data(
                self.name + ".lidar_latent",
                np.random.randn(
                    1,
                    in_channels,
                    config.lidar_resolution_width,
                    config.lidar_resolution_height,
                ).astype(np.float32),
                is_param=True,
            )
            self.tile_latent = F.Tile(self.name + ".tile_latent")

        # Adaptive pooling target sizes
        self.avgpool_img_target_size = (
            config.img_vert_anchors,
            config.img_horz_anchors,
        )

        # Lidar encoder — fully TTSIM ResNet
        self.lidar_encoder = ResNetEncoder_TTSIM(
            f"{name}.lidar_encoder",
            in_channels=in_channels,
            arch=config.lidar_architecture,
        )

        # Global pooling and lidar avgpool - using Resize
        self.global_pool_lidar_target_size = (1, 1)
        self.avgpool_lidar_target_size = (
            config.lidar_vert_anchors,
            config.lidar_horz_anchors,
        )

        lidar_time_frames = [1, 1, 1, 1]
        self.global_pool_img_target_size = (1, 1)

        start_index = 0
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        # Transformer fusion modules
        transformers = []
        for i in range(4):
            transformers.append(
                GPT(
                    f"{self.name}.transformers.{i}",
                    n_embd=self.image_encoder.feature_info.info[start_index + i][
                        "num_chs"
                    ],
                    config=config,
                    lidar_time_frames=lidar_time_frames[i],
                )
            )
        self.transformers = SimNN.ModuleList(transformers)

        # Channel conversion layers — store each op via setattr so __setattr__
        # routes SimOpHandle into _op_hndls (plain list assignment bypasses it)
        for i in range(4):
            img_ch = self.image_encoder.feature_info.info[start_index + i]["num_chs"]
            lidar_ch = self.lidar_encoder.feature_info.info[start_index + i]["num_chs"]

            setattr(
                self,
                f"lidar_channel_to_img_{i}",
                F.Conv2d(
                    f"{self.name}.lidar_channel_to_img.{i}",
                    lidar_ch,
                    img_ch,
                    kernel_size=1,
                ),
            )
            setattr(
                self,
                f"lidar_channel_to_img_bias_{i}",
                F.Bias(
                    f"{self.name}.lidar_channel_to_img_bias.{i}",
                    [1, img_ch, 1, 1],
                ),
            )
            setattr(
                self,
                f"img_channel_to_lidar_{i}",
                F.Conv2d(
                    f"{self.name}.img_channel_to_lidar.{i}",
                    img_ch,
                    lidar_ch,
                    kernel_size=1,
                ),
            )
            setattr(
                self,
                f"img_channel_to_lidar_bias_{i}",
                F.Bias(
                    f"{self.name}.img_channel_to_lidar_bias.{i}",
                    [1, lidar_ch, 1, 1],
                ),
            )

        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3][
            "num_chs"
        ]
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        # Feature fusion configuration
        if self.config.transformer_decoder_join:
            self.num_features = self.lidar_encoder.feature_info.info[start_index + 3][
                "num_chs"
            ]
        else:
            if self.config.add_features:
                self.lidar_to_img_features_end = F.Linear(
                    self.name + ".lidar_to_img_features_end",
                    self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"],
                )
                self.lidar_to_img_features_end_bias = F.Bias(
                    self.name + ".lidar_to_img_features_end_bias",
                    [self.image_encoder.feature_info.info[start_index + 3]["num_chs"]],
                )
                self.num_features = self.image_encoder.feature_info.info[
                    start_index + 3
                ]["num_chs"]
            else:
                self.num_features = (
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
                    + self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
                )

        # FPN fusion layers
        channel = self.config.bev_features_channels
        # Separate relu ops per FPN level to avoid shared-op graph cycles
        self.relu_p5 = F.Relu(self.name + ".relu_p5")
        self.relu_p4 = F.Relu(self.name + ".relu_p4")
        self.relu_p3 = F.Relu(self.name + ".relu_p3")

        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = F.Resize(
                self.name + ".upsample",
                scale_factor=self.config.bev_upsample_factor,
                mode="linear",
                coordinate_transformation_mode="half_pixel",
            )

            self.upsample2_size = (
                self.config.lidar_resolution_height
                // self.config.bev_down_sample_factor,
                self.config.lidar_resolution_width
                // self.config.bev_down_sample_factor,
            )

            self.up_conv5 = F.Conv2d(
                self.name + ".up_conv5", channel, channel, kernel_size=3, padding=1
            )
            self.up_conv5_bias = F.Bias(
                self.name + ".up_conv5_bias",
                [1, channel, 1, 1],
            )
            self.up_conv4 = F.Conv2d(
                self.name + ".up_conv4", channel, channel, kernel_size=3, padding=1
            )
            self.up_conv4_bias = F.Bias(
                self.name + ".up_conv4_bias",
                [1, channel, 1, 1],
            )

            self.c5_conv = F.Conv2d(
                self.name + ".c5_conv",
                self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                channel,
                kernel_size=1,
            )
            self.c5_conv_bias = F.Bias(
                self.name + ".c5_conv_bias",
                [1, channel, 1, 1],
            )

            # Pre-allocate upsample2_dynamic (scale updated at call time)
            self.upsample2_dynamic = F.Resize(
                self.name + ".upsample2_dynamic",
                scale_factor=[1.0, 1.0],
                mode="linear",
                coordinate_transformation_mode="half_pixel",
            )

        # Pre-allocate global fusion ops
        if not self.config.transformer_decoder_join:
            if self.config.add_features:
                self.add_fused = F.Add(self.name + ".add_fused")
            else:
                self.concat_fused = F.ConcatX(self.name + ".concat_fused", axis=1)

        # Pre-allocate per-layer fusion ops (resize with placeholder scale, add ops)
        for i in range(4):
            setattr(
                self,
                f"fuse_img_resize_{i}",
                F.Resize(
                    f"{self.name}.fuse_features.{i}.img_resize",
                    scale_factor=[1.0, 1.0],
                    mode="linear",
                    coordinate_transformation_mode="half_pixel",
                ),
            )
            setattr(
                self,
                f"fuse_lidar_resize_{i}",
                F.Resize(
                    f"{self.name}.fuse_features.{i}.lidar_resize",
                    scale_factor=[1.0, 1.0],
                    mode="linear",
                    coordinate_transformation_mode="half_pixel",
                ),
            )
            setattr(
                self,
                f"fuse_add_img_{i}",
                F.Add(f"{self.name}.fuse_features.{i}.add_img"),
            )
            setattr(
                self,
                f"fuse_add_lidar_{i}",
                F.Add(f"{self.name}.fuse_features.{i}.add_lidar"),
            )

        super().link_op2module()

    def top_down(self, x):
        """FPN top-down pathway."""
        p5 = self.relu_p5(self.c5_conv_bias(self.c5_conv(x)))
        p5_up = self.upsample(p5)
        p4 = self.relu_p4(self.up_conv5_bias(self.up_conv5(p5_up)))

        # Dynamic upsample for specific size (update scale at call time)
        curr_h, curr_w = p4.shape[2], p4.shape[3]
        target_h, target_w = self.upsample2_size
        scale_h = float(target_h) / float(curr_h)
        scale_w = float(target_w) / float(curr_w)
        self.upsample2_dynamic.params[1][1].data = np.array(
            [scale_h, scale_w], dtype=np.float32
        )
        p4_up = self.upsample2_dynamic(p4)
        p4_up.link_module = self  # Set link_module for up_conv4 input
        p3 = self.relu_p3(self.up_conv4_bias(self.up_conv4(p4_up)))

        return p3

    def __call__(self, image, lidar):
        """
        Forward pass with image + LiDAR fusion.

        All operations are pure TTSIM — no PyTorch conversion needed.

        Args:
            image: Input image SimTensor
            lidar: Input LiDAR BEV SimTensor

        Returns:
            features: BEV features (if detect_boxes or use_bev_semantic)
            fused_features: Global fused features
            image_feature_grid: Image feature grid (if use_semantic or use_depth)
        """
        image_features = image
        lidar_features = lidar

        # Use latent representation if configured
        if self.config.latent:
            batch_size = lidar.shape[0]
            repeats = F._from_data(
                self.name + ".tile_repeats",
                np.array([batch_size, 1, 1, 1], dtype=np.int64),
                is_const=True,
            )
            self._tensors[repeats.name] = repeats
            lidar_features = self.tile_latent(self.lidar_latent, repeats)

        # Stem
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.image_encoder.forward_stem(image_features)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.lidar_encoder.forward_stem(lidar_features)

        # Debug dict for per-stage captures
        if not hasattr(self, "_debug"):
            self._debug = {}

        # Process 4 encoder stages with transformer fusion
        for i in range(4):
            image_features = self.image_encoder.forward_stage(i, image_features)
            lidar_features = self.lidar_encoder.forward_stage(i, lidar_features)

            image_features, lidar_features = self.fuse_features(
                image_features, lidar_features, i
            )

        # Store features for outputs
        if self.config.detect_boxes or self.config.use_bev_semantic:
            x4 = lidar_features

        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features

        # Global feature fusion
        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            # Global pooling to (1, 1) using adaptive avg pool
            # NOTE: Cannot use F.Resize with scale 1/N because int(N * (1/N)) can be 0
            # due to floating-point precision (e.g. int(7 * 0.142857) = int(0.999999) = 0)
            image_features = self._adaptive_avg_pool2d(
                image_features, 1, 1, self.name + ".global_pool_img"
            )
            image_features.link_module = self
            image_features = image_features.flatten(start_dim=1)

            lidar_features = self._adaptive_avg_pool2d(
                lidar_features, 1, 1, self.name + ".global_pool_lidar"
            )
            lidar_features.link_module = self
            lidar_features = lidar_features.flatten(start_dim=1)

            if self.config.add_features:
                lidar_features = self.lidar_to_img_features_end_bias(
                    self.lidar_to_img_features_end(lidar_features)
                )
                fused_features = self.add_fused(image_features, lidar_features)
            else:
                fused_features = self.concat_fused(image_features, lidar_features)

        # Generate BEV features if needed
        if self.config.detect_boxes or self.config.use_bev_semantic:
            features = self.top_down(x4)
        else:
            features = None

        return features, fused_features, image_feature_grid

    def _adaptive_avg_pool2d(self, x, target_h, target_w, op_name):
        """
        Emulate nn.AdaptiveAvgPool2d for arbitrary input/output size combinations.

        When input_size >= output_size AND divides evenly, uses F.AveragePool2d.
        Otherwise, falls back to numpy computation matching PyTorch's exact
        adaptive pool formula:
            start_index[i] = floor(i * input_size / output_size)
            end_index[i]   = ceil((i+1) * input_size / output_size)
        """
        in_h, in_w = x.shape[2], x.shape[3]

        # Check if simple pooling works: input >= output and divides evenly
        if (
            in_h >= target_h
            and in_w >= target_w
            and in_h % target_h == 0
            and in_w % target_w == 0
        ):
            kh, kw = in_h // target_h, in_w // target_w
            pool = F.AveragePool2d(
                op_name,
                kernel_shape=[kh, kw],
                strides=[kh, kw],
            )
            pool.link_module = self
            # Store as attribute so get_ops() can discover it
            attr_name = "_pool_" + op_name.replace(".", "_")
            setattr(self, attr_name, pool)
            result = pool(x)
            return result

        # Fallback: numpy-based adaptive avg pool matching PyTorch exactly
        import math

        if hasattr(x, "data") and x.data is not None:
            input_data = x.data  # shape: (N, C, H, W)
        else:
            # Shape inference only, create zeros
            input_data = np.zeros(x.shape, dtype=np.float32)

        N, C = input_data.shape[0], input_data.shape[1]
        output_data = np.zeros((N, C, target_h, target_w), dtype=np.float32)

        for oh in range(target_h):
            h_start = math.floor(oh * in_h / target_h)
            h_end = math.ceil((oh + 1) * in_h / target_h)
            for ow in range(target_w):
                w_start = math.floor(ow * in_w / target_w)
                w_end = math.ceil((ow + 1) * in_w / target_w)
                output_data[:, :, oh, ow] = input_data[
                    :, :, h_start:h_end, w_start:w_end
                ].mean(axis=(2, 3))

        result = F._from_data(op_name, output_data)
        self._tensors[result.name] = result
        return result

    def fuse_features(self, image_features, lidar_features, layer_idx):
        """
        Transformer-based feature fusion at one scale.

        Args:
            image_features: Image branch features (SimTensor)
            lidar_features: LiDAR branch features (SimTensor)
            layer_idx: Fusion layer index (0-3)

        Returns:
            Fused image and lidar features (SimTensor)
        """
        # Downsample to anchor grid using AdaptiveAvgPool2d
        target_h, target_w = self.avgpool_img_target_size
        image_embd_layer = self._adaptive_avg_pool2d(
            image_features,
            target_h,
            target_w,
            f"{self.name}.fuse_features.{layer_idx}.img_avgpool",
        )

        target_h, target_w = self.avgpool_lidar_target_size
        lidar_embd_layer = self._adaptive_avg_pool2d(
            lidar_features,
            target_h,
            target_w,
            f"{self.name}.fuse_features.{layer_idx}.lidar_avgpool",
        )

        # Channel alignment
        lidar_embd_layer = getattr(self, f"lidar_channel_to_img_{layer_idx}")(
            lidar_embd_layer
        )
        lidar_embd_layer = getattr(self, f"lidar_channel_to_img_bias_{layer_idx}")(
            lidar_embd_layer
        )
        lidar_embd_layer.link_module = self  # Set link_module for GPT input
        image_embd_layer.link_module = self  # Set link_module for GPT input (permute)

        # Transformer fusion
        image_features_layer, lidar_features_layer = self.transformers[layer_idx](
            image_embd_layer, lidar_embd_layer
        )

        lidar_features_layer = getattr(self, f"img_channel_to_lidar_{layer_idx}")(
            lidar_features_layer
        )
        lidar_features_layer = getattr(self, f"img_channel_to_lidar_bias_{layer_idx}")(
            lidar_features_layer
        )
        lidar_features_layer.link_module = self  # Set link_module for Resize input

        # Upsample back to original resolution (bilinear interpolation)
        img_h, img_w = image_features.shape[2], image_features.shape[3]
        img_scale_h = float(img_h) / float(image_features_layer.shape[2])
        img_scale_w = float(img_w) / float(image_features_layer.shape[3])

        img_resize_op = getattr(self, f"fuse_img_resize_{layer_idx}")
        img_resize_op.params[1][1].data = np.array(
            [img_scale_h, img_scale_w], dtype=np.float32
        )
        image_features_layer = img_resize_op(image_features_layer)

        lidar_h, lidar_w = lidar_features.shape[2], lidar_features.shape[3]
        lidar_scale_h = float(lidar_h) / float(lidar_features_layer.shape[2])
        lidar_scale_w = float(lidar_w) / float(lidar_features_layer.shape[3])

        lidar_resize_op = getattr(self, f"fuse_lidar_resize_{layer_idx}")
        lidar_resize_op.params[1][1].data = np.array(
            [lidar_scale_h, lidar_scale_w], dtype=np.float32
        )
        lidar_features_layer = lidar_resize_op(lidar_features_layer)

        # Residual connections
        add_img_op = getattr(self, f"fuse_add_img_{layer_idx}")
        image_features = add_img_op(image_features, image_features_layer)

        add_lidar_op = getattr(self, f"fuse_add_lidar_{layer_idx}")
        lidar_features = add_lidar_op(lidar_features, lidar_features_layer)

        return image_features, lidar_features


def run_standalone(outdir: str = ".") -> None:
    """
    Example usage of TransfuserBackbone TTSIM conversion.
    """
    
    logger.info("=" * 60)
    logger.info("TransfuserBackbone TTSIM Conversion")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Usage:")
    logger.info(
        " from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig"
    )
    logger.info(" config = TransfuserConfig(...)")
    logger.info(" model = TransfuserBackbone('transfuser', config)")
    logger.info(" features, fused_features, img_grid = model(image, lidar)")
    logger.info("")
    logger.info("Notes:")
    logger.info(" - Custom layers fully converted to ttsim")
    logger.info(" - timm encoders kept as PyTorch (pretrained models)")
    logger.info(" - Preserves exact logic, inputs, and outputs")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_standalone()
