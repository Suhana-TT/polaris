#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim Motion Forecasting Head for UniAD.

For each tracked agent the head predicts K anchor-based trajectory modes
over `predict_steps` future time steps using a multi-layer transformer decoder.

Architecture:
  - Per-agent query feature from track head
  - Anchor embedding lookup
  - Transformer decoder (query = agent query + anchor emb, memory = BEV)
  - Regression MLP -> [K, predict_steps, 2]  (dx, dy per step)
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class _MotionDecLayer(SimNN.Module):
    """Single motion decoder layer (cross-attention to BEV)."""

    def __init__(self, name, embed_dims, num_heads, ffn_dim, bs, sq, bev_len):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        nH = num_heads
        dH = embed_dims // num_heads
        E = embed_dims
        sk_sa = sq  # self-attn: keys/values from query itself
        sk_ca = bev_len  # cross-attn: keys/values from BEV

        # Self-attention among agent-anchor queries
        self.wq_sa = SimNN.Linear(name + ".wq_sa", embed_dims, embed_dims)
        self.wk_sa = SimNN.Linear(name + ".wk_sa", embed_dims, embed_dims)
        self.wv_sa = SimNN.Linear(name + ".wv_sa", embed_dims, embed_dims)
        self.wo_sa = SimNN.Linear(name + ".wo_sa", embed_dims, embed_dims)
        self.sm_sa = F.Softmax(name + ".sm_sa", axis=-1)
        self.ln_sa = F.LayerNorm(name + ".ln_sa", embed_dims)

        # Cross-attention to BEV
        self.wq_ca = SimNN.Linear(name + ".wq_ca", embed_dims, embed_dims)
        self.wk_ca = SimNN.Linear(name + ".wk_ca", embed_dims, embed_dims)
        self.wv_ca = SimNN.Linear(name + ".wv_ca", embed_dims, embed_dims)
        self.wo_ca = SimNN.Linear(name + ".wo_ca", embed_dims, embed_dims)
        self.sm_ca = F.Softmax(name + ".sm_ca", axis=-1)
        self.ln_ca = F.LayerNorm(name + ".ln_ca", embed_dims)

        # FFN
        self.ff1 = SimNN.Linear(name + ".ff1", embed_dims, ffn_dim)
        self.relu = F.Relu(name + ".relu")
        self.ff2 = SimNN.Linear(name + ".ff2", ffn_dim, embed_dims)
        self.ln_ff = F.LayerNorm(name + ".ln_ff", embed_dims)

        # Residual adds
        self.add_sa = F.Add(name + ".add_sa")
        self.add_ca = F.Add(name + ".add_ca")
        self.add_ff = F.Add(name + ".add_ff")

        # SA ops (sq x sq attention)
        self.sa_qk_mm = F.MatMul(name + ".sa_qk_mm")
        self.sa_sc_mul = F.Mul(name + ".sa_sc_mul")
        self.sa_av_mm = F.MatMul(name + ".sa_av_mm")
        self.sa_rq = F.Reshape(name + ".sa_rq")
        self.sa_tq = F.Transpose(name + ".sa_tq", perm=[0, 2, 1, 3])
        self.sa_rk = F.Reshape(name + ".sa_rk")
        self.sa_tk = F.Transpose(name + ".sa_tk", perm=[0, 2, 1, 3])
        self.sa_tk2 = F.Transpose(name + ".sa_tk2", perm=[0, 1, 3, 2])
        self.sa_rv = F.Reshape(name + ".sa_rv")
        self.sa_tv = F.Transpose(name + ".sa_tv", perm=[0, 2, 1, 3])
        self.sa_to = F.Transpose(name + ".sa_to", perm=[0, 2, 1, 3])
        self.sa_ro = F.Reshape(name + ".sa_ro")

        # CA ops (sq x bev_len attention)
        self.ca_qk_mm = F.MatMul(name + ".ca_qk_mm")
        self.ca_sc_mul = F.Mul(name + ".ca_sc_mul")
        self.ca_av_mm = F.MatMul(name + ".ca_av_mm")
        self.ca_rq = F.Reshape(name + ".ca_rq")
        self.ca_tq = F.Transpose(name + ".ca_tq", perm=[0, 2, 1, 3])
        self.ca_rk = F.Reshape(name + ".ca_rk")
        self.ca_tk = F.Transpose(name + ".ca_tk", perm=[0, 2, 1, 3])
        self.ca_tk2 = F.Transpose(name + ".ca_tk2", perm=[0, 1, 3, 2])
        self.ca_rv = F.Reshape(name + ".ca_rv")
        self.ca_tv = F.Transpose(name + ".ca_tv", perm=[0, 2, 1, 3])
        self.ca_to = F.Transpose(name + ".ca_to", perm=[0, 2, 1, 3])
        self.ca_ro = F.Reshape(name + ".ca_ro")

        # Shape constants
        self._sa_shQ = F._from_data(
            name + ".sa_shQ", np.array([bs, sq, nH, dH], dtype=np.int64), is_const=True
        )
        self._sa_shK = F._from_data(
            name + ".sa_shK",
            np.array([bs, sk_sa, nH, dH], dtype=np.int64),
            is_const=True,
        )
        self._sa_shV = F._from_data(
            name + ".sa_shV",
            np.array([bs, sk_sa, nH, dH], dtype=np.int64),
            is_const=True,
        )
        self._sa_shO = F._from_data(
            name + ".sa_shO", np.array([bs, sq, E], dtype=np.int64), is_const=True
        )
        self._sa_sc = F._from_data(
            name + ".sa_sc", np.float32(1.0 / (dH**0.5)), is_const=True
        )
        self._ca_shQ = F._from_data(
            name + ".ca_shQ", np.array([bs, sq, nH, dH], dtype=np.int64), is_const=True
        )
        self._ca_shK = F._from_data(
            name + ".ca_shK",
            np.array([bs, sk_ca, nH, dH], dtype=np.int64),
            is_const=True,
        )
        self._ca_shV = F._from_data(
            name + ".ca_shV",
            np.array([bs, sk_ca, nH, dH], dtype=np.int64),
            is_const=True,
        )
        self._ca_shO = F._from_data(
            name + ".ca_shO", np.array([bs, sq, E], dtype=np.int64), is_const=True
        )
        self._ca_sc = F._from_data(
            name + ".ca_sc", np.float32(1.0 / (dH**0.5)), is_const=True
        )

        super().link_op2module()

    def __call__(self, query, bev):
        # Self-attention
        Q = self.sa_tq(self.sa_rq(self.wq_sa(query), self._sa_shQ))
        K = self.sa_tk2(self.sa_tk(self.sa_rk(self.wk_sa(query), self._sa_shK)))
        V = self.sa_tv(self.sa_rv(self.wv_sa(query), self._sa_shV))
        sa = self.sa_ro(
            self.sa_to(
                self.sa_av_mm(
                    self.sm_sa(self.sa_sc_mul(self.sa_qk_mm(Q, K), self._sa_sc)), V
                )
            ),
            self._sa_shO,
        )
        query = self.ln_sa(self.add_sa(query, self.wo_sa(sa)))

        # Cross-attention to BEV
        Q = self.ca_tq(self.ca_rq(self.wq_ca(query), self._ca_shQ))
        K = self.ca_tk2(self.ca_tk(self.ca_rk(self.wk_ca(bev), self._ca_shK)))
        V = self.ca_tv(self.ca_rv(self.wv_ca(bev), self._ca_shV))
        ca = self.ca_ro(
            self.ca_to(
                self.ca_av_mm(
                    self.sm_ca(self.ca_sc_mul(self.ca_qk_mm(Q, K), self._ca_sc)), V
                )
            ),
            self._ca_shO,
        )
        query = self.ln_ca(self.add_ca(query, self.wo_ca(ca)))

        # FFN
        ff = self.ff2(self.relu(self.ff1(query)))
        return self.ln_ff(self.add_ff(query, ff))


class MotionHead(SimNN.Module):
    """
    Motion forecasting head.

    Args (from cfg):
        embed_dims, num_query, predict_steps, num_anchor
        num_dec_layers, num_heads, ffn_dim
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name
        embed_dims = cfg.get("embed_dims", 256)
        num_query = cfg.get("num_query", 900)
        predict_steps = cfg.get("predict_steps", 12)
        num_anchor = cfg.get("num_anchor", 6)
        num_dec_layers = cfg.get("num_dec_layers", 6)
        num_heads = cfg.get("num_heads", 8)
        ffn_dim = cfg.get("ffn_dim", 512)
        self.bs = cfg.get("bs", 1)
        self.num_query = num_query
        self.predict_steps = predict_steps
        self.num_anchor = num_anchor
        self.embed_dims = embed_dims
        self._num_dec = num_dec_layers

        # Anchor embedding
        self.anchor_emb = F.Embedding(name + ".anchor_emb", num_anchor, embed_dims)

        # Project track query to motion query
        self.query_proj = SimNN.Linear(name + ".q_proj", embed_dims, embed_dims)

        # Fuse query + anchor (both projected to embed_dims then added)
        self.anc_proj = SimNN.Linear(name + ".anc_proj", embed_dims, embed_dims)
        self.fuse_add = F.Add(name + ".fuse_add")

        # Representative decoder layer
        nq_K = num_query * num_anchor
        bev_h = cfg.get("bev_h", 50)
        bev_w = cfg.get("bev_w", 50)
        self._dec_layer = _MotionDecLayer(
            name + ".dec", embed_dims, num_heads, ffn_dim, self.bs, nq_K, bev_h * bev_w
        )

        # Trajectory regression head
        self.traj_head_fc1 = SimNN.Linear(name + ".traj_fc1", embed_dims, embed_dims)
        self.traj_relu = F.Relu(name + ".traj_relu")
        self.traj_head_fc2 = SimNN.Linear(
            name + ".traj_fc2", embed_dims, predict_steps * 2
        )

        # Pre-register inline reshape/tile ops used in __call__
        bs_ = self.bs
        nq_ = num_query
        K_ = num_anchor
        E_ = embed_dims
        self._q_tile = F.Reshape(name + ".q_tile")
        self._q_tile2 = F.Tile(name + ".q_tile2")
        self._q_tile3 = F.Reshape(name + ".q_tile3")
        self._a_tile = F.Tile(name + ".a_tile")
        self._traj_rs = F.Reshape(name + ".traj_reshape")
        # Pre-register shape/rep constants
        self._q_tile_shape = F._from_data(
            name + ".q_tile_shape",
            np.array([bs_ * nq_, 1, E_], dtype=np.int64),
            is_const=True,
        )
        self._q_tile2_reps = F._from_data(
            name + ".q_tile2_reps", np.array([1, K_, 1], dtype=np.int64), is_const=True
        )
        self._q_tile3_shape = F._from_data(
            name + ".q_tile3_shape",
            np.array([bs_, nq_ * K_, E_], dtype=np.int64),
            is_const=True,
        )
        self._a_tile_reps = F._from_data(
            name + ".a_tile_reps", np.array([1, nq_, 1], dtype=np.int64), is_const=True
        )
        self._traj_shape = F._from_data(
            name + ".traj_shape",
            np.array([bs_, nq_, K_, predict_steps, 2], dtype=np.int64),
            is_const=True,
        )

        super().link_op2module()

        # Pre-register anchor index tensor
        self._anc_idx = F._from_shape(
            name + ".anc_idx", [self.bs, num_anchor], np_dtype=np.int64
        )
        self._tensors[self._anc_idx.name] = self._anc_idx

    def __call__(self, bev_feat, track_out: dict):
        """
        bev_feat  : [bs, bev_h*bev_w, embed_dims]
        track_out : dict with 'query_feats' [bs, num_query, embed_dims]
        Returns dict with 'traj_preds' [bs, num_query, num_anchor, predict_steps, 2]
        """
        bs = self.bs
        query_feat = track_out["query_feats"]  # [bs, nq, E]

        # Project query feature
        query = self.query_proj(query_feat)  # [bs, nq, E]

        # Anchor embeddings: [bs, K, E]
        anc_emb = self.anchor_emb(self._anc_idx)  # [bs, K, E]
        anc_proj = self.anc_proj(anc_emb)  # [bs, K, E]

        # We build combined [bs, nq*K, E] by repeating query nq times per anchor
        # and repeating anchor nq times for each query, then adding.
        # To avoid 4D broadcast issues, we tile each to [bs, nq*K, E]:
        #   query_tiled: each of nq queries repeated K times  -> [bs, nq*K, E]
        #   anc_tiled:   each of K anchors repeated nq times  -> [bs, nq*K, E]
        nq = self.num_query
        K = self.num_anchor
        E = self.embed_dims

        query_tiled = self._q_tile(query, self._q_tile_shape)
        query_tiled = self._q_tile2(query_tiled, self._q_tile2_reps)  # [bs*nq, K, E]
        query_tiled = self._q_tile3(query_tiled, self._q_tile3_shape)  # [bs, nq*K, E]

        anc_tiled = self._a_tile(anc_proj, self._a_tile_reps)  # [bs, nq*K, E]

        combined = self.fuse_add(query_tiled, anc_tiled)  # [bs, nq*K, E]

        # Decoder
        hs = self._dec_layer(combined, bev_feat)

        # Set repeat count
        dec_ops: dict = {}
        self._dec_layer.get_ops(dec_ops)
        for _, op_obj in dec_ops.items():
            op_obj.repeat_count = self._num_dec

        # Traj regression: [bs, nq*K, predict_steps*2]
        traj_flat = self.traj_head_fc2(self.traj_relu(self.traj_head_fc1(hs)))

        # Reshape to [bs, nq, K, predict_steps, 2]
        traj = self._traj_rs(traj_flat, self._traj_shape)

        return {
            "traj_preds": traj,
            "motion_feats": hs,
        }
