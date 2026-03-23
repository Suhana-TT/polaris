#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim Panoptic Segmentation head for UniAD (PansegformerHead).

Simplified architecture:
  - "Thing" branch: object queries cross-attend to BEV, output mask logits
  - "Stuff" branch: stuff queries cross-attend to BEV, output mask logits
  - Final: bilinear upsample to canvas size
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class _QueryDecoder(SimNN.Module):
    """Simple cross-attention decoder for seg queries."""

    def __init__(self, name, embed_dims, num_heads, num_layers, num_query, bs, bev_len):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.num_layers = num_layers
        self.num_query = num_query
        nH = num_heads
        dH = embed_dims // num_heads
        E = embed_dims
        sq = num_query
        sk = bev_len

        self.query_emb = F.Embedding(name + ".q_emb", num_query, embed_dims)

        # Representative cross-attn layer
        self.wq = SimNN.Linear(name + ".wq", embed_dims, embed_dims)
        self.wk = SimNN.Linear(name + ".wk", embed_dims, embed_dims)
        self.wv = SimNN.Linear(name + ".wv", embed_dims, embed_dims)
        self.wo = SimNN.Linear(name + ".wo", embed_dims, embed_dims)
        self.sm = F.Softmax(name + ".sm", axis=-1)
        self.ln = F.LayerNorm(name + ".ln", embed_dims)

        self.ff1 = SimNN.Linear(name + ".ff1", embed_dims, embed_dims * 2)
        self.relu = F.Relu(name + ".relu")
        self.ff2 = SimNN.Linear(name + ".ff2", embed_dims * 2, embed_dims)
        self.ln_ff = F.LayerNorm(name + ".ln_ff", embed_dims)
        self.add_attn = F.Add(name + ".add_attn")
        self.add_ffn = F.Add(name + ".add_ffn")
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

        self._q_idx = F._from_shape(name + ".q_idx", [bs, num_query], np_dtype=np.int64)
        self._tensors[self._q_idx.name] = self._q_idx

    def __call__(self, bev, bs):
        query = self.query_emb(self._q_idx)  # [bs, nq, embed_dims]

        Q = self.tq(self.rq(self.wq(query), self._shQ))
        K = self.tk2(self.tk(self.rk(self.wk(bev), self._shK)))
        V = self.tv(self.rv(self.wv(bev), self._shV))
        out = self.ro(
            self.to(self.av_mm(self.sm(self.sc_mul(self.qk_mm(Q, K), self._sc)), V)),
            self._shO,
        )
        out = self.wo(out)
        query = self.ln(self.add_attn(query, out))

        ffn_out = self.ff2(self.relu(self.ff1(query)))
        query = self.ln_ff(self.add_ffn(query, ffn_out))

        # Repeat count for all ops in this decoder
        layer_ops: dict = {}
        self.get_ops(layer_ops)
        for _, op_obj in layer_ops.items():
            op_obj.repeat_count = self.num_layers

        return query


class PansegformerHead(SimNN.Module):
    """
    Panoptic segmentation head.

    Args (from cfg):
        embed_dims, bev_h, bev_w
        num_things_classes, num_stuff_classes
        num_dec_things, num_dec_stuff
        num_query_things, num_query_stuff
        canvas_h, canvas_w  (output resolution)
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name

        embed_dims = cfg.get("embed_dims", 256)
        bev_h = cfg.get("bev_h", 50)
        bev_w = cfg.get("bev_w", 50)
        num_heads = cfg.get("num_heads", 8)
        num_things = cfg.get("num_things_classes", 8)
        num_stuff = cfg.get("num_stuff_classes", 2)
        num_dec_things = cfg.get("num_dec_things", 6)
        num_dec_stuff = cfg.get("num_dec_stuff", 6)
        num_q_things = cfg.get("num_query_things", 300)
        num_q_stuff = cfg.get("num_query_stuff", 100)
        canvas_h = cfg.get("canvas_h", bev_h * 4)
        canvas_w = cfg.get("canvas_w", bev_w * 4)
        self.bs = cfg.get("bs", 1)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.embed_dims = embed_dims

        # Thing decoder
        self.thing_dec = _QueryDecoder(
            name + ".thing_dec",
            embed_dims,
            num_heads,
            num_dec_things,
            num_q_things,
            self.bs,
            bev_h * bev_w,
        )
        self.thing_cls = SimNN.Linear(name + ".thing_cls", embed_dims, num_things)

        # Stuff decoder
        self.stuff_dec = _QueryDecoder(
            name + ".stuff_dec",
            embed_dims,
            num_heads,
            num_dec_stuff,
            num_q_stuff,
            self.bs,
            bev_h * bev_w,
        )
        self.stuff_cls = SimNN.Linear(name + ".stuff_cls", embed_dims, num_stuff)

        # Mask projection: [embed_dims] x [bev_h*bev_w, embed_dims] -> [num_query, bev_h*bev_w]
        self.mask_proj = SimNN.Linear(name + ".mask_proj", embed_dims, bev_h * bev_w)

        super().link_op2module()

    def __call__(self, bev_feat):
        """
        bev_feat : [bs, bev_h*bev_w, embed_dims]
        Returns dict with 'seg_logits'
        """
        bs = self.bs

        # Thing branch
        thing_feats = self.thing_dec(bev_feat, bs)  # [bs, nq_t, embed_dims]
        thing_cls = self.thing_cls(thing_feats)  # [bs, nq_t, num_things]
        thing_masks = self.mask_proj(thing_feats)  # [bs, nq_t, bev_h*bev_w]

        # Stuff branch
        stuff_feats = self.stuff_dec(bev_feat, bs)
        stuff_cls = self.stuff_cls(stuff_feats)
        stuff_masks = self.mask_proj(stuff_feats)  # shares weights intentionally

        return {
            "thing_cls": thing_cls,
            "thing_masks": thing_masks,
            "stuff_cls": stuff_cls,
            "stuff_masks": stuff_masks,
        }
