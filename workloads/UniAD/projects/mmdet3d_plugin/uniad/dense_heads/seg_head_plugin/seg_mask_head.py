# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: seg_head_plugin/seg_mask_head.py — SimNN replacements.
No torch, no mmcv, no einops imports.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from ....ttsim_utils import TRANSFORMER  # type: ignore[import-not-found]


class Mlp(SimNN.Module):
    """TTSim SimNN MLP block."""

    def __init__(
        self, name, in_features, hidden_features=None, out_features=None, drop=0.0
    ):
        super().__init__()
        self.name = name
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = SimNN.Linear(name + ".fc1", in_features, hidden_features)
        self.act = F.Relu(name + ".act")
        self.fc2 = SimNN.Linear(name + ".fc2", hidden_features, out_features)
        self.drop = F.Dropout(name + ".drop", drop, True)
        super().link_op2module()

    def __call__(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(SimNN.Module):
    """TTSim SimNN self-attention block."""

    def __init__(self, name, dim, num_heads=2, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.name = name
        self.num_heads = num_heads
        self.qkv = SimNN.Linear(name + ".qkv", dim, dim * 3)
        self.attn_drop = F.Dropout(name + ".attn_drop", attn_drop, True)
        self.proj = SimNN.Linear(name + ".proj", dim, dim)
        self.proj_drop = F.Dropout(name + ".proj_drop", proj_drop, True)
        self.softmax = F.Softmax(name + ".softmax", axis=-1)
        super().link_op2module()

    def __call__(self, x):
        x = self.proj_drop(self.proj(self.attn_drop(self.softmax(self.qkv(x)))))
        return x


class Attention(SimNN.Module):
    """TTSim SimNN cross-attention block."""

    def __init__(self, name, dim, num_heads=2, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.name = name
        self.num_heads = num_heads
        self.q = SimNN.Linear(name + ".q", dim, dim)
        self.k = SimNN.Linear(name + ".k", dim, dim)
        self.v = SimNN.Linear(name + ".v", dim, dim)
        self.attn_drop = F.Dropout(name + ".attn_drop", attn_drop, True)
        self.proj = SimNN.Linear(name + ".proj", dim, dim)
        self.proj_drop = F.Dropout(name + ".proj_drop", proj_drop, True)
        self.softmax = F.Softmax(name + ".softmax", axis=-1)
        self.linear_l1 = SimNN.Linear(name + ".linear_l1", num_heads, num_heads)
        self.linear_l1_relu = F.Relu(name + ".linear_l1_relu")
        self.linear = SimNN.Linear(name + ".linear", num_heads, 1)
        self.linear_relu = F.Relu(name + ".linear_relu")
        super().link_op2module()

    def __call__(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        attn = self.softmax(self.attn_drop(q))
        x = self.proj_drop(self.proj(attn))
        mask = self.linear_relu(self.linear(self.linear_l1_relu(self.linear_l1(attn))))
        return x, mask


class AttentionTail(SimNN.Module):
    """TTSim SimNN attention tail block."""

    def __init__(self, name, dim, num_heads=2, attn_drop=0.0):
        super().__init__()
        self.name = name
        self.num_heads = num_heads
        self.q = SimNN.Linear(name + ".q", dim, dim)
        self.k = SimNN.Linear(name + ".k", dim, dim)
        self.softmax = F.Softmax(name + ".softmax", axis=-1)
        self.linear_l1 = SimNN.Linear(name + ".linear_l1", num_heads, num_heads)
        self.linear_l1_relu = F.Relu(name + ".linear_l1_relu")
        self.linear = SimNN.Linear(name + ".linear", num_heads, 1)
        self.linear_relu = F.Relu(name + ".linear_relu")
        super().link_op2module()

    def __call__(self, query, key, key_padding_mask=None, hw_lvl=None):
        q = self.q(query)
        k = self.k(key)
        attn = self.softmax(q)
        mask = self.linear_relu(self.linear(self.linear_l1_relu(self.linear_l1(attn))))
        return mask


class DropPath(SimNN.Module):
    """TTSim SimNN DropPath (no-op in inference)."""

    def __init__(self, name, drop_prob=0.0):
        super().__init__()
        self.name = name
        self.drop = F.Dropout(name + ".drop", drop_prob, True)
        super().link_op2module()

    def __call__(self, x):
        return x  # inference: no drop path applied


class Block(SimNN.Module):
    """TTSim SimNN transformer block."""

    def __init__(
        self,
        name,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        self_attn=False,
    ):
        super().__init__()
        self.name = name
        self.self_attn_flag = self_attn
        self.head_norm1 = F.LayerNorm(name + ".head_norm1", dim)
        self.head_norm2 = F.LayerNorm(name + ".head_norm2", dim)
        self.attn = Attention(
            name + ".attn",
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path_op = (
            DropPath(name + ".drop_path", drop_path) if drop_path > 0.0 else None
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(name + ".mlp", dim, mlp_hidden_dim, dim, drop=drop)
        self.add1 = F.Add(name + ".add1")
        self.add2 = F.Add(name + ".add2")
        if self_attn:
            self.self_attention = SelfAttention(
                name + ".self_attn",
                dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.norm3 = F.LayerNorm(name + ".norm3", dim)
            self.add_sa = F.Add(name + ".add_sa")
        super().link_op2module()

    def __call__(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        if self.self_attn_flag:
            sa_out = self.self_attention(query)
            query = self.norm3(self.add_sa(query, sa_out))
        x, mask = self.attn(query, key, value, key_padding_mask, hw_lvl=hw_lvl)
        drop_x = x if self.drop_path_op is None else self.drop_path_op(x)
        query = self.head_norm1(self.add1(query, drop_x))
        mlp_out = self.mlp(query)
        drop_mlp = mlp_out if self.drop_path_op is None else self.drop_path_op(mlp_out)
        query = self.head_norm2(self.add2(query, drop_mlp))
        return query, mask


@TRANSFORMER.register_module()
class SegMaskHead(SimNN.Module):
    """TTSim SimNN SegMaskHead."""

    def __init__(
        self,
        cfg=None,
        d_model=16,
        nhead=2,
        num_encoder_layers=6,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        self_attn=False,
    ):
        super().__init__()
        self.name = "seg_mask_head"
        self._blocks_list = []
        for i in range(num_decoder_layers):
            blk = Block(
                self.name + f".block{i}",
                dim=d_model,
                num_heads=nhead,
                mlp_ratio=4.0,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                self_attn=self_attn,
            )
            self._blocks_list.append(blk)
            setattr(self, f"block{i}", blk)
        self.attnen = AttentionTail(
            self.name + ".attnen", d_model, num_heads=nhead, attn_drop=0.0
        )
        super().link_op2module()

    def __call__(
        self,
        memory,
        mask_memory,
        pos_memory,
        query_embed,
        mask_query,
        pos_query,
        hw_lvl,
    ):
        masks = []
        inter_query = []
        for blk in self._blocks_list:
            query_embed, mask = blk(
                query_embed, memory, memory, key_padding_mask=mask_memory, hw_lvl=hw_lvl
            )
            masks.append(mask)
            inter_query.append(query_embed)
        attn = self.attnen(
            query_embed, memory, key_padding_mask=mask_memory, hw_lvl=hw_lvl
        )
        return attn, masks, inter_query
