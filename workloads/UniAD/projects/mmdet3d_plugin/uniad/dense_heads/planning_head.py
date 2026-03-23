#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim Planning Head for UniAD (PlanningHeadSingleMode).

Architecture:
  1. Navigation command embedding (3 commands: left/straight/right)
  2. Fuse: [BEV_summary, motion_summary, nav_emb] via MLP
  3. Transformer decoder: planning query attends to BEV
  4. Regression MLP -> planning_steps * 2  (dx, dy per step)
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class _PlanDecLayer(SimNN.Module):
    """Single planning decoder layer (cross-attention to BEV)."""

    def __init__(self, name, embed_dims, num_heads, ffn_dim, bs, bev_len):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        nH = num_heads
        dH = embed_dims // num_heads
        E = embed_dims
        sq = 1  # planning query is always a single token
        sk = bev_len

        self.wq = SimNN.Linear(name + ".wq", embed_dims, embed_dims)
        self.wk = SimNN.Linear(name + ".wk", embed_dims, embed_dims)
        self.wv = SimNN.Linear(name + ".wv", embed_dims, embed_dims)
        self.wo = SimNN.Linear(name + ".wo", embed_dims, embed_dims)
        self.sm = F.Softmax(name + ".sm", axis=-1)
        self.ln = F.LayerNorm(name + ".ln", embed_dims)
        self.add_res = F.Add(name + ".add_res")
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

        self.ff1 = SimNN.Linear(name + ".ff1", embed_dims, ffn_dim)
        self.relu = F.Relu(name + ".relu")
        self.ff2 = SimNN.Linear(name + ".ff2", ffn_dim, embed_dims)
        self.ln_ff = F.LayerNorm(name + ".ln_ff", embed_dims)
        self.add_ff = F.Add(name + ".add_ff")

        super().link_op2module()

    def __call__(self, query, memory):
        Q = self.tq(self.rq(self.wq(query), self._shQ))
        K = self.tk2(self.tk(self.rk(self.wk(memory), self._shK)))
        V = self.tv(self.rv(self.wv(memory), self._shV))
        out = self.ro(
            self.to(self.av_mm(self.sm(self.sc_mul(self.qk_mm(Q, K), self._sc)), V)),
            self._shO,
        )
        out = self.wo(out)
        query = self.ln(self.add_res(query, out))

        ff = self.ff2(self.relu(self.ff1(query)))
        return self.ln_ff(self.add_ff(query, ff))


class PlanningHead(SimNN.Module):
    """
    Ego-vehicle planning head.

    Args (from cfg):
        embed_dims, planning_steps, num_dec_layers, num_heads, ffn_dim, bs
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name
        embed_dims = cfg.get("embed_dims", 256)
        planning_steps = cfg.get("planning_steps", 6)
        num_dec_layers = cfg.get("num_dec_layers", 3)
        num_heads = cfg.get("num_heads", 8)
        ffn_dim = cfg.get("ffn_dim", 512)
        self.bs = cfg.get("bs", 1)
        bev_h = cfg.get("bev_h", 50)
        bev_w = cfg.get("bev_w", 50)
        self.embed_dims = embed_dims
        self.planning_steps = planning_steps
        self._num_dec = num_dec_layers

        # Navigation command embedding (3 commands)
        self.navi_emb = F.Embedding(name + ".navi_emb", 3, embed_dims)

        # Position embedding for the single planning token
        self.pos_emb = F.Embedding(name + ".pos_emb", 1, embed_dims)
        self.pos_add = F.Add(name + ".pos_add")

        # MLP fuser: [BEV_summary, motion_summary, nav_emb] -> embed_dims
        # Input: [bs, 1, 3*embed_dims] -> output: [bs, 1, embed_dims]
        fuser_in = embed_dims * 3
        self.fuser_fc1 = SimNN.Linear(name + ".fuser_fc1", fuser_in, embed_dims)
        self.fuser_ln = F.LayerNorm(name + ".fuser_ln", embed_dims)
        self.fuser_relu = F.Relu(name + ".fuser_relu")

        # Summary projections: project [bs, S, E] -> [bs, 1, E]
        # We use AdaptiveAvgPool1d (reduces seq_len to 1)
        self.bev_pool = F.AdaptiveAvgPool1d(
            name + ".bev_pool", adaptive=True, output_size=1
        )
        self.motion_pool = F.AdaptiveAvgPool1d(
            name + ".motion_pool", adaptive=True, output_size=1
        )

        # Decoder
        self._dec_layer = _PlanDecLayer(
            name + ".dec", embed_dims, num_heads, ffn_dim, self.bs, bev_h * bev_w
        )

        # Regression MLP: embed_dims -> planning_steps*2
        self.reg_fc1 = SimNN.Linear(name + ".reg_fc1", embed_dims, embed_dims)
        self.reg_relu = F.Relu(name + ".reg_relu")
        self.reg_fc2 = SimNN.Linear(name + ".reg_fc2", embed_dims, planning_steps * 2)

        # Pre-register inline ops used in __call__
        self._bev_t = F.Transpose(name + ".bev_t", perm=[0, 2, 1])
        self._bev_t2 = F.Transpose(name + ".bev_t2", perm=[0, 2, 1])
        self._mot_t = F.Transpose(name + ".mot_t", perm=[0, 2, 1])
        self._mot_t2 = F.Transpose(name + ".mot_t2", perm=[0, 2, 1])
        self._fuse_cat = F.ConcatX(name + ".fuse_cat", axis=2)
        self._traj_rs = F.Reshape(name + ".traj_reshape")
        # Pre-register traj shape constant
        bs_ = cfg.get("bs", 1)
        self._traj_shape = F._from_data(
            name + ".traj_shape",
            np.array([bs_, planning_steps, 2], dtype=np.int64),
            is_const=True,
        )

        super().link_op2module()

        # Pre-register index tensors for embedding lookups
        bs_ = cfg.get("bs", 1)
        self._nav_idx = F._from_shape(name + ".nav_idx", [bs_, 1], np_dtype=np.int64)
        self._tensors[self._nav_idx.name] = self._nav_idx
        self._pos_idx = F._from_shape(name + ".pos_idx", [bs_, 1], np_dtype=np.int64)
        self._tensors[self._pos_idx.name] = self._pos_idx

    def __call__(self, bev_feat, motion_out: dict, occ_out: dict | None = None):
        """
        bev_feat   : [bs, bev_h*bev_w, embed_dims]
        motion_out : dict from MotionHead with 'motion_feats' [bs, nq*K, embed_dims]
        Returns dict with 'plan_traj' [bs, planning_steps, 2]
        """
        bs = self.bs
        E = self.embed_dims

        # ── BEV summary via AdaptiveAvgPool1d ─────────────────────────────
        bev_t = self._bev_t(bev_feat)
        bev_pooled = self.bev_pool(bev_t)
        bev_summary = self._bev_t2(bev_pooled)  # [bs, 1, E]

        # ── Motion summary ────────────────────────────────────────────────
        motion_feats = motion_out.get("motion_feats", bev_feat)
        mot_t = self._mot_t(motion_feats)
        mot_pooled = self.motion_pool(mot_t)
        motion_summary = self._mot_t2(mot_pooled)  # [bs, 1, E]

        # ── Navigation embedding ──────────────────────────────────────────
        nav_emb = self.navi_emb(self._nav_idx)  # [bs, 1, E]

        # ── Fuse ──────────────────────────────────────────────────────────
        fused = self._fuse_cat(bev_summary, motion_summary, nav_emb)
        plan_query = self.fuser_relu(self.fuser_ln(self.fuser_fc1(fused)))  # [bs, 1, E]

        # ── Add positional embedding ──────────────────────────────────────
        plan_query = self.pos_add(plan_query, self.pos_emb(self._pos_idx))

        # ── Decoder ───────────────────────────────────────────────────────
        hs = self._dec_layer(plan_query, bev_feat)  # [bs, 1, E]

        dec_ops: dict = {}
        self._dec_layer.get_ops(dec_ops)
        for _, op_obj in dec_ops.items():
            op_obj.repeat_count = self._num_dec

        # ── Regression ────────────────────────────────────────────────────
        traj_flat = self.reg_fc2(
            self.reg_relu(self.reg_fc1(hs))
        )  # [bs, 1, planning_steps*2]
        traj = self._traj_rs(traj_flat, self._traj_shape)

        return {"plan_traj": traj}
