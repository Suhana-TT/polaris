#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim BEVFormer track detection head for UniAD.

Architecture:
  - Object query embedding
  - Cross-attention decoder (query x BEV memory)
  - Classification branch: (Linear -> LN -> ReLU) x n -> Linear
  - Regression branch:     (Linear -> ReLU) x n -> Linear(code_size)
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# ─── cls branch sub-modules ──────────────────────────────────────────────────


class _ClsLayer(SimNN.Module):
    """One intermediate cls layer: Linear -> LN -> ReLU."""

    def __init__(self, name, embed_dims):
        super().__init__()
        self.name = name
        self.fc = SimNN.Linear(name + ".fc", embed_dims, embed_dims)
        self.ln = F.LayerNorm(name + ".ln", embed_dims)
        self.relu = F.Relu(name + ".relu")
        super().link_op2module()

    def __call__(self, x):
        return self.relu(self.ln(self.fc(x)))


class _ClsHead(SimNN.Module):
    """Classification branch: (Linear -> LN -> ReLU) x num_fcs -> Linear."""

    def __init__(self, name, embed_dims, num_classes, num_fcs=2):
        super().__init__()
        self.name = name
        self.mid_layers = SimNN.ModuleList(
            [_ClsLayer(f"{name}.mid{i}", embed_dims) for i in range(num_fcs)]
        )
        self.final = SimNN.Linear(f"{name}.final", embed_dims, num_classes)
        super().link_op2module()

    def __call__(self, x):
        for layer in self.mid_layers:
            x = layer(x)
        return self.final(x)


# ─── reg branch sub-modules ──────────────────────────────────────────────────


class _RegLayer(SimNN.Module):
    """One intermediate reg layer: Linear -> ReLU."""

    def __init__(self, name, embed_dims):
        super().__init__()
        self.name = name
        self.fc = SimNN.Linear(name + ".fc", embed_dims, embed_dims)
        self.relu = F.Relu(name + ".relu")
        super().link_op2module()

    def __call__(self, x):
        return self.relu(self.fc(x))


class _RegHead(SimNN.Module):
    """Regression branch: (Linear -> ReLU) x num_fcs -> Linear(code_size)."""

    def __init__(self, name, embed_dims, code_size, num_fcs=2):
        super().__init__()
        self.name = name
        self.mid_layers = SimNN.ModuleList(
            [_RegLayer(f"{name}.mid{i}", embed_dims) for i in range(num_fcs)]
        )
        self.final = SimNN.Linear(f"{name}.final", embed_dims, code_size)
        super().link_op2module()

    def __call__(self, x):
        for layer in self.mid_layers:
            x = layer(x)
        return self.final(x)


# ─── attention sub-layers ────────────────────────────────────────────────────


class _AttnLayer(SimNN.Module):
    """Multi-head attention (self or cross)."""

    def __init__(self, name, embed_dims, num_heads, bs, sq, sk):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        nH = num_heads
        dH = embed_dims // num_heads
        E = embed_dims
        self.wq = SimNN.Linear(name + ".wq", embed_dims, embed_dims)
        self.wk = SimNN.Linear(name + ".wk", embed_dims, embed_dims)
        self.wv = SimNN.Linear(name + ".wv", embed_dims, embed_dims)
        self.wo = SimNN.Linear(name + ".wo", embed_dims, embed_dims)
        self.sm = F.Softmax(name + ".sm", axis=-1)
        self.ln = F.LayerNorm(name + ".ln", embed_dims)
        self.add = F.Add(name + ".add")
        self.qk_mm = F.MatMul(name + ".qk_mm")
        self.sc_mul = F.Mul(name + ".sc_mul")
        self.av_mm = F.MatMul(name + ".av_mm")
        self.rq = F.Reshape(name + ".rq")
        self.tq = F.Transpose(name + ".tq", perm=[0, 2, 1, 3])
        self.rk = F.Reshape(name + ".rk")
        self.tk = F.Transpose(name + ".tk", perm=[0, 2, 1, 3])
        self.tk2 = F.Transpose(name + ".tk2", perm=[0, 1, 3, 2])
        self.rv = F.Reshape(name + ".rv")
        self.tv = F.Transpose(name + ".tv", perm=[0, 2, 1, 3])
        self.to = F.Transpose(name + ".to", perm=[0, 2, 1, 3])
        self.ro = F.Reshape(name + ".ro")
        self._shQ = F._from_data(
            name + ".shQ", np.array([bs, sq, nH, dH], dtype=np.int64), is_const=True
        )
        self._shK = F._from_data(
            name + ".shK", np.array([bs, sk, nH, dH], dtype=np.int64), is_const=True
        )
        self._shV = F._from_data(
            name + ".shV", np.array([bs, sk, nH, dH], dtype=np.int64), is_const=True
        )
        self._shO = F._from_data(
            name + ".shO", np.array([bs, sq, E], dtype=np.int64), is_const=True
        )
        self._sc = F._from_data(
            name + ".sc", np.float32(1.0 / (dH**0.5)), is_const=True
        )
        super().link_op2module()

    def __call__(self, q, kv):
        Q = self.tq(self.rq(self.wq(q), self._shQ))
        K = self.tk2(self.tk(self.rk(self.wk(kv), self._shK)))
        V = self.tv(self.rv(self.wv(kv), self._shV))
        out = self.ro(
            self.to(self.av_mm(self.sm(self.sc_mul(self.qk_mm(Q, K), self._sc)), V)),
            self._shO,
        )
        return self.ln(self.add(q, self.wo(out)))


class _FFNLayer(SimNN.Module):
    def __init__(self, name, embed_dims, ffn_dim):
        super().__init__()
        self.name = name
        self.fc1 = SimNN.Linear(name + ".fc1", embed_dims, ffn_dim)
        self.relu = F.Relu(name + ".relu")
        self.fc2 = SimNN.Linear(name + ".fc2", ffn_dim, embed_dims)
        self.ln = F.LayerNorm(name + ".ln", embed_dims)
        self.add_res = F.Add(name + ".add_res")
        super().link_op2module()

    def __call__(self, x):
        return self.ln(self.add_res(x, self.fc2(self.relu(self.fc1(x)))))


class _DecoderLayer(SimNN.Module):
    """Single transformer decoder layer (self-attn + cross-attn + FFN)."""

    def __init__(self, name, embed_dims, num_heads, ffn_dim, bs, nq, bev_len):
        super().__init__()
        self.name = name
        self.sa = _AttnLayer(name + ".sa", embed_dims, num_heads, bs, nq, nq)
        self.ca = _AttnLayer(name + ".ca", embed_dims, num_heads, bs, nq, bev_len)
        self.ffn = _FFNLayer(name + ".ffn", embed_dims, ffn_dim)
        super().link_op2module()

    def __call__(self, query, memory):
        query = self.sa(query, query)
        query = self.ca(query, memory)
        return self.ffn(query)


# ─── main head ───────────────────────────────────────────────────────────────


class BEVFormerTrackHead(SimNN.Module):
    """
    Detection / tracking head for UniAD.

    Args (from cfg):
        embed_dims, num_query, num_classes, code_size,
        num_dec_layers, num_heads, ffn_dim, num_cls_fcs, num_reg_fcs, bs
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name
        embed_dims = cfg.get("embed_dims", 256)
        num_query = cfg.get("num_query", 900)
        num_classes = cfg.get("num_classes", 10)
        code_size = cfg.get("code_size", 10)
        num_dec_layers = cfg.get("num_dec_layers", 6)
        num_heads = cfg.get("num_heads", 8)
        ffn_dim = cfg.get("ffn_dim", 512)
        num_cls_fcs = cfg.get("num_cls_fcs", 2)
        num_reg_fcs = cfg.get("num_reg_fcs", 2)
        self.bs = cfg.get("bs", 1)
        bev_h = cfg.get("bev_h", 50)
        bev_w = cfg.get("bev_w", 50)
        self.num_query = num_query
        self.embed_dims = embed_dims
        self._num_dec = num_dec_layers

        # Object query embedding: 2*embed_dims (content + position)
        self.query_emb = F.Embedding(name + ".query_emb", num_query, embed_dims * 2)

        # Representative decoder layer (repeat_count = num_dec_layers)
        self._dec_layer = _DecoderLayer(
            name + ".dec",
            embed_dims,
            num_heads,
            ffn_dim,
            self.bs,
            num_query,
            bev_h * bev_w,
        )

        # Prediction heads
        self.cls_head = _ClsHead(name + ".cls", embed_dims, num_classes, num_cls_fcs)
        self.reg_head = _RegHead(name + ".reg", embed_dims, code_size, num_reg_fcs)

        bs_ = cfg.get("bs", 1)
        # Pre-register SliceF for content query extraction
        self._q_slice = F.SliceF(
            name + ".q_content", out_shape=[bs_, num_query, embed_dims]
        )
        # Pre-register slice const tensors
        self._q_start = F._from_data(
            name + ".q_start", np.array([0], dtype=np.int64), is_const=True
        )
        self._q_end = F._from_data(
            name + ".q_end", np.array([embed_dims], dtype=np.int64), is_const=True
        )
        self._q_axis = F._from_data(
            name + ".q_axis", np.array([2], dtype=np.int64), is_const=True
        )
        self._q_step = F._from_data(
            name + ".q_step", np.array([1], dtype=np.int64), is_const=True
        )

        super().link_op2module()

        # Pre-register query index tensor
        self._q_idx = F._from_shape(
            name + ".q_idx", [bs_, num_query], is_param=False, np_dtype=np.int64
        )
        self._tensors[self._q_idx.name] = self._q_idx

    def __call__(self, bev_feat):
        """
        bev_feat : [bs, bev_h*bev_w, embed_dims]
        Returns dict with query_feats, cls_scores, bbox_preds
        """
        bs = self.bs

        # Object query lookup
        q_emb = self.query_emb(self._q_idx)  # [bs, num_query, 2*embed_dims]

        # Take content half via slice
        query = self._q_slice(
            q_emb, self._q_start, self._q_end, self._q_axis, self._q_step
        )

        # Decoder
        hs = self._dec_layer(query, bev_feat)

        # Apply repeat_count to decoder ops
        dec_ops: dict = {}
        self._dec_layer.get_ops(dec_ops)
        for _, op_obj in dec_ops.items():
            op_obj.repeat_count = self._num_dec

        cls_scores = self.cls_head(hs)
        bbox_preds = self.reg_head(hs)

        return {
            "query_feats": hs,
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
        }
