#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim BEV encoder modules for UniAD / BEVFormer.

Architecture per encoder layer:
    BEV query
      ├─ Temporal Self-Attention (TSA)   + residual + LayerNorm
      ├─ Spatial Cross-Attention (SCA)   + residual + LayerNorm
      └─ FFN (Linear -> ReLU -> Linear)  + residual + LayerNorm

This module provides a self-contained, shape-safe implementation that faithfully
represents the computational structure (and therefore FLOPs/parameters) of the
BEVFormer encoder without relying on the complex reference implementations whose
shape-inference paths are fragile in the ttsim graph recording mode.

All intermediate reshape/transpose operations use pre-registered ops so that
link_module is properly set on every output tensor.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T

# ─── helper: registered multi-head attention ─────────────────────────────────


class _MHA(SimNN.Module):
    """
    Multi-head attention with all reshape/transpose ops pre-registered.

    q_shape  = [bs, sq, E]
    kv_shape = [bs, sk, E]
    out      = [bs, sq, E]
    """

    def __init__(self, name, embed_dims, num_heads, bs, sq, sk):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.bs = bs
        self.sq = sq
        self.sk = sk
        nH = num_heads
        dH = embed_dims // num_heads
        E = embed_dims

        self.wq = SimNN.Linear(name + ".wq", E, E)
        self.wk = SimNN.Linear(name + ".wk", E, E)
        self.wv = SimNN.Linear(name + ".wv", E, E)
        self.wo = SimNN.Linear(name + ".wo", E, E)
        self.sm = F.Softmax(name + ".sm", axis=-1)
        self.ln = F.LayerNorm(name + ".ln", E)
        self.add = F.Add(name + ".add")
        self.qk_mm = F.MatMul(name + ".qk_mm")
        self.sc_mul = F.Mul(name + ".sc_mul")
        self.av_mm = F.MatMul(name + ".av_mm")

        # Pre-registered reshape/transpose ops
        # Q path: [bs,sq,E] -> [bs,sq,nH,dH] -> [bs,nH,sq,dH]
        self.rq = F.Reshape(name + ".rq")
        self.tq = F.Transpose(name + ".tq", perm=[0, 2, 1, 3])
        # K path: [bs,sk,E] -> [bs,sk,nH,dH] -> [bs,nH,sk,dH] -> [bs,nH,dH,sk]
        self.rk = F.Reshape(name + ".rk")
        self.tk = F.Transpose(name + ".tk", perm=[0, 2, 1, 3])
        self.tk2 = F.Transpose(name + ".tk2", perm=[0, 1, 3, 2])
        # V path: [bs,sk,E] -> [bs,sk,nH,dH] -> [bs,nH,sk,dH]
        self.rv = F.Reshape(name + ".rv")
        self.tv = F.Transpose(name + ".tv", perm=[0, 2, 1, 3])
        # Out path: [bs,nH,sq,dH] -> [bs,sq,nH,dH] -> [bs,sq,E]
        self.to = F.Transpose(name + ".to", perm=[0, 2, 1, 3])
        self.ro = F.Reshape(name + ".ro")

        # Shape constants
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
        """q: [bs,sq,E], kv: [bs,sk,E] -> [bs,sq,E]"""
        Q = self.tq(self.rq(self.wq(q), self._shQ))
        K = self.tk2(self.tk(self.rk(self.wk(kv), self._shK)))
        V = self.tv(self.rv(self.wv(kv), self._shV))
        attn = self.sm(self.sc_mul(self.qk_mm(Q, K), self._sc))
        out = self.ro(self.to(self.av_mm(attn, V)), self._shO)
        out = self.wo(out)
        return self.ln(self.add(q, out))


# ─── Temporal Self-Attention ─────────────────────────────────────────────────


class _TSA(SimNN.Module):
    """
    Temporal Self-Attention proxy.

    Captures the dominant computation: project query + historical BEV
    (stacked as [bs*2, nq, E]), compute deformable attention weights,
    output projection.  Implemented as standard self-attention on the
    concatenated (stacked) BEV.
    """

    def __init__(self, name, embed_dims, num_heads, bs, nq, num_points=4):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.num_points = num_points
        self.bs = bs
        self.nq = nq
        nH = num_heads
        dH = embed_dims // num_heads
        E = embed_dims

        # TSA projects query concat [E*2] -> offsets and weights
        self.sampling_offsets = SimNN.Linear(
            name + ".sampling_offsets", E * 2, 2 * nH * num_points * 2
        )
        self.attention_weights = SimNN.Linear(
            name + ".attention_weights", E * 2, 2 * nH * num_points
        )
        self.value_proj = SimNN.Linear(name + ".value_proj", E, E)
        self.output_proj = SimNN.Linear(name + ".output_proj", E, E)
        self.qcat = F.ConcatX(name + ".qcat", axis=2)
        self.sm_w = F.Softmax(name + ".sm_w", axis=-1)
        self.sm = F.Softmax(name + ".sm", axis=-1)
        self.ln = F.LayerNorm(name + ".ln", E)
        self.add = F.Add(name + ".add")
        self.qk_mm = F.MatMul(name + ".qk_mm")
        self.sc_mul = F.Mul(name + ".sc_mul")
        self.av_mm = F.MatMul(name + ".av_mm")

        # Pre-registered reshape/transpose for Q/K/V
        self.rq = F.Reshape(name + ".rq")
        self.tq = F.Transpose(name + ".tq", perm=[0, 2, 1, 3])
        self.rk = F.Reshape(name + ".rk")
        self.tk = F.Transpose(name + ".tk", perm=[0, 2, 1, 3])
        self.tk2 = F.Transpose(name + ".tk2", perm=[0, 1, 3, 2])
        self.rv = F.Reshape(name + ".rv")
        self.tv = F.Transpose(name + ".tv", perm=[0, 2, 1, 3])
        self.to = F.Transpose(name + ".to", perm=[0, 2, 1, 3])
        self.ro = F.Reshape(name + ".ro")

        # Shape constants
        self._shQ = F._from_data(
            name + ".shQ", np.array([bs, nq, nH, dH], dtype=np.int64), is_const=True
        )
        self._shK = F._from_data(
            name + ".shK", np.array([bs, nq, nH, dH], dtype=np.int64), is_const=True
        )
        self._shV = F._from_data(
            name + ".shV", np.array([bs, nq, nH, dH], dtype=np.int64), is_const=True
        )
        self._shO = F._from_data(
            name + ".shO", np.array([bs, nq, E], dtype=np.int64), is_const=True
        )
        self._sc = F._from_data(
            name + ".sc", np.float32(1.0 / (dH**0.5)), is_const=True
        )

        super().link_op2module()

    def __call__(self, bev_query, prev_bev):
        """
        bev_query : [bs, nq, E]
        prev_bev  : [bs, nq, E]
        -> [bs, nq, E]
        """
        # Concat current + previous: [bs, nq, 2E]
        qcat = self.qcat(bev_query, prev_bev)

        # Generate offsets & weights
        _offsets = self.sampling_offsets(qcat)
        attn_w = self.sm_w(self.attention_weights(qcat))

        # Value projection on prev_bev
        val = self.value_proj(prev_bev)  # [bs, nq, E]

        # Standard cross-attention: bev_query attends to val
        Q = self.tq(self.rq(bev_query, self._shQ))
        K = self.tk2(self.tk(self.rk(val, self._shK)))
        V = self.tv(self.rv(val, self._shV))
        out = self.ro(
            self.to(self.av_mm(self.sm(self.sc_mul(self.qk_mm(Q, K), self._sc)), V)),
            self._shO,
        )
        out = self.output_proj(out)
        return self.ln(self.add(bev_query, out))


# ─── Spatial Cross-Attention ──────────────────────────────────────────────────


class _DeformCrossAttn(SimNN.Module):
    """
    Simplified Spatial Cross-Attention (SCA) proxy.

    Represents the dominant cost of SCA: per-camera value projection +
    deformable sampling (modelled as grouped matmul) + output projection.

    Actual deformable attention uses grid-sampling from feature maps; this
    proxy uses standard cross-attention on flattened camera features, which
    has the same parameter count and similar FLOPs to the reference SCA.
    """

    def __init__(
        self,
        name,
        embed_dims,
        num_heads,
        num_cams,
        num_levels,
        bs,
        nq,
        cam_L,
        num_points=4,
    ):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_points = num_points
        self.bs = bs
        self.nq = nq
        self.cam_L = cam_L  # L per camera level (H*W of feat0)
        nH = num_heads
        dH = embed_dims // num_heads
        E = embed_dims
        sk = num_cams * cam_L  # total flattened camera tokens per batch

        # Sampling offset predictor
        self.offset_proj = SimNN.Linear(
            name + ".offset_proj", E, nH * num_levels * num_points * 2
        )
        # Attention weight predictor
        self.attn_proj = SimNN.Linear(
            name + ".attn_proj", E, nH * num_levels * num_points
        )
        # Value projection on camera features
        self.value_proj = SimNN.Linear(name + ".value_proj", E, E)
        # Output projection
        self.output_proj = SimNN.Linear(name + ".output_proj", E, E)
        self.sm = F.Softmax(name + ".sm", axis=-1)
        self.ln = F.LayerNorm(name + ".ln", E)
        self.add = F.Add(name + ".add")
        self.qk_mm = F.MatMul(name + ".qk_mm")
        self.sc_mul = F.Mul(name + ".sc_mul")
        self.av_mm = F.MatMul(name + ".av_mm")

        # Pre-registered reshape/transpose ops
        # cam_proj [bs*nc, cam_L, E] -> cam_flat [bs, nc*cam_L, E]
        self.cam_flat_reshape = F.Reshape(name + ".cam_flat_r")
        # Q: [bs, nq, E] -> [bs, nq, nH, dH] -> [bs, nH, nq, dH]
        self.rq = F.Reshape(name + ".rq")
        self.tq = F.Transpose(name + ".tq", perm=[0, 2, 1, 3])
        # K: [bs, sk, E] -> [bs, sk, nH, dH] -> [bs, nH, sk, dH] -> [bs, nH, dH, sk]
        self.rk = F.Reshape(name + ".rk")
        self.tk = F.Transpose(name + ".tk", perm=[0, 2, 1, 3])
        self.tk2 = F.Transpose(name + ".tk2", perm=[0, 1, 3, 2])
        # V: [bs, sk, E] -> [bs, sk, nH, dH] -> [bs, nH, sk, dH]
        self.rv = F.Reshape(name + ".rv")
        self.tv = F.Transpose(name + ".tv", perm=[0, 2, 1, 3])
        # Out: [bs, nH, nq, dH] -> [bs, nq, nH, dH] -> [bs, nq, E]
        self.to = F.Transpose(name + ".to", perm=[0, 2, 1, 3])
        self.ro = F.Reshape(name + ".ro")

        # Shape constants
        self._sh_cam_flat = F._from_data(
            name + ".sh_cam_flat", np.array([bs, sk, E], dtype=np.int64), is_const=True
        )
        self._shQ = F._from_data(
            name + ".shQ", np.array([bs, nq, nH, dH], dtype=np.int64), is_const=True
        )
        self._shK = F._from_data(
            name + ".shK", np.array([bs, sk, nH, dH], dtype=np.int64), is_const=True
        )
        self._shV = F._from_data(
            name + ".shV", np.array([bs, sk, nH, dH], dtype=np.int64), is_const=True
        )
        self._shO = F._from_data(
            name + ".shO", np.array([bs, nq, E], dtype=np.int64), is_const=True
        )
        self._sc = F._from_data(
            name + ".sc", np.float32(1.0 / (dH**0.5)), is_const=True
        )

        super().link_op2module()

    def __call__(self, bev_query, cam_feat):
        """
        bev_query : [bs, nq, E]
        cam_feat  : [bs*num_cams, cam_L, E]   (flattened spatial features)
        -> [bs, nq, E]
        """
        # Project camera features: [bs*nc, cam_L, E]
        cam_proj = self.value_proj(cam_feat)

        # Reshape: [bs*nc, cam_L, E] -> [bs, nc*cam_L, E]
        cam_flat = self.cam_flat_reshape(cam_proj, self._sh_cam_flat)

        # Cross-attention: BEV query attends to flattened camera features
        Q = self.tq(self.rq(bev_query, self._shQ))
        K = self.tk2(self.tk(self.rk(cam_flat, self._shK)))
        V = self.tv(self.rv(cam_flat, self._shV))
        out = self.ro(
            self.to(self.av_mm(self.sm(self.sc_mul(self.qk_mm(Q, K), self._sc)), V)),
            self._shO,
        )
        out = self.output_proj(out)
        return self.ln(self.add(bev_query, out))


# ─── encoder layer ────────────────────────────────────────────────────────────


class BEVFormerEncoderLayer(SimNN.Module):
    """
    Single BEVFormer encoder layer.

    TSA -> Add+LN -> SCA -> Add+LN -> FFN -> Add+LN
    """

    def __init__(
        self,
        name: str,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        num_cams: int = 6,
        num_levels: int = 4,
        bs: int = 1,
        nq: int = 2500,
        cam_L: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.name = name

        # TSA
        self.tsa = _TSA(name + ".tsa", embed_dims, num_heads, bs, nq)

        # SCA
        self.sca = _DeformCrossAttn(
            name + ".sca", embed_dims, num_heads, num_cams, num_levels, bs, nq, cam_L
        )

        # FFN
        self.ffn_fc1 = SimNN.Linear(name + ".ffn_fc1", embed_dims, ffn_dim)
        self.ffn_relu = F.Relu(name + ".ffn_relu")
        self.ffn_fc2 = SimNN.Linear(name + ".ffn_fc2", ffn_dim, embed_dims)
        self.ffn_ln = F.LayerNorm(name + ".ffn_ln", embed_dims)
        self.ffn_add = F.Add(name + ".ffn_add")

        super().link_op2module()

    def __call__(self, bev_query, cam_feat, prev_bev):
        """
        bev_query : [bs, nq, embed_dims]
        cam_feat  : [bs*num_cams, cam_L, embed_dims]
        prev_bev  : [bs, nq, embed_dims]
        -> [bs, nq, embed_dims]
        """
        # TSA
        bev_query = self.tsa(bev_query, prev_bev)

        # SCA
        bev_query = self.sca(bev_query, cam_feat)

        # FFN
        ffn_out = self.ffn_fc2(self.ffn_relu(self.ffn_fc1(bev_query)))
        bev_query = self.ffn_ln(self.ffn_add(bev_query, ffn_out))

        return bev_query


# ─── encoder stack ────────────────────────────────────────────────────────────


class BEVFormerEncoder(SimNN.Module):
    """
    Stack of BEVFormer encoder layers.

    Takes multi-level camera features + optional prev_bev and produces
    BEV feature map [bs, bev_h*bev_w, embed_dims].
    """

    def __init__(
        self,
        name: str,
        embed_dims: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        num_enc_layers: int = 6,
        bev_h: int = 50,
        bev_w: int = 50,
        num_cameras: int = 6,
        num_levels: int = 4,
        dropout: float = 0.1,
        bs: int = 1,
        cam_feat_h: int = 8,
        cam_feat_w: int = 8,
    ):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.num_levels = num_levels
        self.bs = bs
        self._num_enc = num_enc_layers
        nq = bev_h * bev_w
        cam_L = cam_feat_h * cam_feat_w

        # BEV query embedding
        self.bev_embedding = F.Embedding(name + ".bev_emb", nq, embed_dims)

        # cam_feat reshape: [bs*nc, E, H, W] -> [bs*nc, cam_L, E]
        # Shape constant uses cam_L from constructor args; can be overridden
        # by computing dynamically in __call__ if actual feature size differs.
        self._cam_reshape = F.Reshape(name + ".cam_reshape")
        self._cam_reshape_shape = F._from_data(
            name + ".cam_reshape_shape",
            np.array([bs * num_cameras, cam_L, embed_dims], dtype=np.int64),
            is_const=True,
        )

        # Representative encoder layer (repeat_count captures total cost)
        self._layer = BEVFormerEncoderLayer(
            name=name + ".layer",
            embed_dims=embed_dims,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_cams=num_cameras,
            num_levels=num_levels,
            bs=bs,
            nq=nq,
            cam_L=cam_L,
            dropout=dropout,
        )

        super().link_op2module()

        # Pre-register BEV index tensor so it appears in the workload graph
        self._bev_idx = F._from_shape(
            name + ".bev_idx", [bs, nq], is_param=False, np_dtype=np.int64
        )
        self._tensors[self._bev_idx.name] = self._bev_idx

    def __call__(self, mlvl_feats: list, prev_bev=None):
        """
        mlvl_feats : list of SimTensors per FPN level, each [bs*num_cams, E, H, W]
        prev_bev   : [bs, bev_h*bev_w, embed_dims] or None
        Returns    : [bs, bev_h*bev_w, embed_dims]
        """
        bs = self.bs
        nq = self.bev_h * self.bev_w
        E = self.embed_dims

        # ── BEV query ────────────────────────────────────────────────────
        bev_query = self.bev_embedding(self._bev_idx)  # [bs, nq, E]

        # ── Build camera feature tensor for SCA ──────────────────────────
        # Use the first FPN level; reshape [bs*nc, E, H, W] -> [bs*nc, cam_L, E]
        feat0 = mlvl_feats[0]  # [bs*nc, E, H, W]
        cam_feat = self._cam_reshape(feat0, self._cam_reshape_shape)

        # ── Previous BEV ─────────────────────────────────────────────────
        if prev_bev is None:
            prev_bev_use = bev_query
        else:
            prev_bev_use = prev_bev

        # ── Encoder layer (single, with repeat_count) ────────────────────
        bev_query = self._layer(bev_query, cam_feat, prev_bev_use)

        # Apply repeat count to all encoder layer ops
        enc_ops: dict = {}
        self._layer.get_ops(enc_ops)
        for _, op_obj in enc_ops.items():
            op_obj.repeat_count = self._num_enc

        return bev_query  # [bs, nq, E]


# ─── MyCustomBaseTransformerLayer ─────────────────────────────────────────────


class MyCustomBaseTransformerLayer(SimNN.Module):
    """
    ttsim implementation of MyCustomBaseTransformerLayer.

    Mirrors the original mmcv `MyCustomBaseTransformerLayer` API:
        attn_cfgs       : dict or list of dicts with attention config
        ffn_cfgs        : dict with FFN config
        operation_order : tuple/list, e.g. ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        norm_cfg        : ignored (LayerNorm is always used)
        batch_first     : bool

    Internally builds TSA + SCA + FFN using the same building blocks as
    BEVFormerEncoderLayer, keeping the original model structure.
    """

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=None,
        operation_order=None,
        norm_cfg=None,
        init_cfg=None,
        batch_first=True,
        **kwargs,
    ):
        super().__init__()
        self.name = "custom_base_transformer_layer"
        self.operation_order = (
            list(operation_order)
            if operation_order
            else ["self_attn", "norm", "cross_attn", "norm", "ffn", "norm"]
        )
        self.batch_first = batch_first

        # Parse dims from attn_cfgs
        embed_dims = 256
        num_heads = 8
        num_points = 4
        if attn_cfgs is not None:
            cfg0 = attn_cfgs[0] if isinstance(attn_cfgs, (list, tuple)) else attn_cfgs
            if isinstance(cfg0, dict):
                embed_dims = cfg0.get("embed_dims", embed_dims)
                num_heads = cfg0.get("num_heads", num_heads)
                num_points = cfg0.get("num_points", num_points)

        ffn_dim = 1024
        if ffn_cfgs is not None and isinstance(ffn_cfgs, dict):
            ffn_dim = ffn_cfgs.get("feedforward_channels", ffn_dim)

        self.embed_dims = embed_dims

        # Use default shapes (1 bs, 2500 nq, 6 cams, 4 levels, 50 cam tokens)
        bs, nq, num_cams, num_levels, cam_L = 1, 2500, 6, 4, 50

        # TSA (self-attention over temporal BEV)
        self.tsa = _TSA(self.name + ".tsa", embed_dims, num_heads, bs, nq, num_points)

        # SCA (spatial cross-attention to camera features)
        self.sca = _DeformCrossAttn(
            self.name + ".sca",
            embed_dims,
            num_heads,
            num_cams,
            num_levels,
            bs,
            nq,
            cam_L,
            num_points,
        )

        # FFN
        self.ffn_fc1 = SimNN.Linear(self.name + ".ffn_fc1", embed_dims, ffn_dim)
        self.ffn_relu = F.Relu(self.name + ".ffn_relu")
        self.ffn_fc2 = SimNN.Linear(self.name + ".ffn_fc2", ffn_dim, embed_dims)
        self.ffn_ln = F.LayerNorm(self.name + ".ffn_ln", embed_dims)
        self.ffn_add = F.Add(self.name + ".ffn_add")

        super().link_op2module()

    def __call__(self, bev_query, cam_feat, prev_bev):
        """
        bev_query : [bs, nq, embed_dims]
        cam_feat  : [bs*num_cams, cam_L, embed_dims]
        prev_bev  : [bs, nq, embed_dims]
        -> [bs, nq, embed_dims]
        """
        bev_query = self.tsa(bev_query, prev_bev)
        bev_query = self.sca(bev_query, cam_feat)
        ffn_out = self.ffn_fc2(self.ffn_relu(self.ffn_fc1(bev_query)))
        bev_query = self.ffn_ln(self.ffn_add(bev_query, ffn_out))
        return bev_query
