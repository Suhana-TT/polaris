#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim Occupancy Prediction Head for UniAD.

Produces a per-cell occupancy probability map over n_future time steps.

Architecture (simplified from OccHead):
  1. BEV feature projection via Conv2d
  2. Down-scale encoder (2x Bottleneck each stride-2)
  3. Future prediction: lightweight transformer decoder + upsampling
  4. Final Conv2d to predict occupancy classes
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class _ConvBnRelu(SimNN.Module):
    def __init__(self, name, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.name = name
        self.conv = F.Conv2d(name + ".conv", in_ch, out_ch, k, stride=s, padding=p)
        self.bn = F.BatchNorm2d(name + ".bn", out_ch)
        self.relu = F.Relu(name + ".relu")
        super().link_op2module()

    def __call__(self, x):
        return self.relu(self.bn(self.conv(x)))


class _OccBottleneck(SimNN.Module):
    """Lightweight bottleneck for occ encoder (stride-2 down-scale)."""

    def __init__(self, name, in_ch, mid_ch=None):
        super().__init__()
        self.name = name
        if mid_ch is None:
            mid_ch = in_ch // 2
        out_ch = in_ch
        self.c1 = F.Conv2d(name + ".c1", in_ch, mid_ch, 3, stride=2, padding=1)
        self.bn1 = F.BatchNorm2d(name + ".bn1", mid_ch)
        self.relu = F.Relu(name + ".relu")
        self.c2 = F.Conv2d(name + ".c2", mid_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = F.BatchNorm2d(name + ".bn2", out_ch)
        # down-sample shortcut
        self.c_ds = F.Conv2d(name + ".c_ds", in_ch, out_ch, 1, stride=2, padding=0)
        self.bn_ds = F.BatchNorm2d(name + ".bn_ds", out_ch)
        self.relu2 = F.Relu(name + ".relu2")
        self.add = F.Add(name + ".add")
        super().link_op2module()

    def __call__(self, x):
        res = self.bn_ds(self.c_ds(x))
        y = self.relu(self.bn1(self.c1(x)))
        y = self.bn2(self.c2(y))
        return self.relu2(self.add(y, res))


class OccHead(SimNN.Module):
    """
    Occupancy prediction head.

    Args (from cfg):
        embed_dims, bev_h, bev_w, n_future, num_occ_classes
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name
        embed_dims = cfg.get("embed_dims", 256)
        bev_h = cfg.get("bev_h", 50)
        bev_w = cfg.get("bev_w", 50)
        n_future = cfg.get("n_future", 4)
        proj_dim = cfg.get("bev_proj_dim", 64)
        num_occ_classes = cfg.get("num_occ_classes", 2)  # free / occupied
        self.bs = cfg.get("bs", 1)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.n_future = n_future
        self.embed_dims = embed_dims
        self.proj_dim = proj_dim

        # BEV feature projection
        self.bev_proj = _ConvBnRelu(
            name + ".bev_proj", embed_dims, proj_dim, k=1, s=1, p=0
        )

        # Down-scale encoder (x2 stride-2 bottlenecks)
        self.ds1 = _OccBottleneck(name + ".ds1", proj_dim)
        self.ds2 = _OccBottleneck(name + ".ds2", proj_dim)

        # Future query: one embedding per future step
        self.future_emb = F.Embedding(name + ".future_emb", n_future + 1, proj_dim)

        # Cross-attention: future query attends to down-scaled BEV
        self.wq = SimNN.Linear(name + ".wq", proj_dim, proj_dim)
        self.wk = SimNN.Linear(name + ".wk", proj_dim, proj_dim)
        self.wv = SimNN.Linear(name + ".wv", proj_dim, proj_dim)
        self.wo = SimNN.Linear(name + ".wo", proj_dim, proj_dim)
        self.sm = F.Softmax(name + ".sm", axis=-1)
        self.ln = F.LayerNorm(name + ".ln", proj_dim)

        # Up-sample back to bev_h x bev_w
        self.up1 = F.ConvTranspose2d(
            name + ".up1", proj_dim, proj_dim, kernel_size=2, stride=2
        )
        self.up2 = F.ConvTranspose2d(
            name + ".up2", proj_dim, proj_dim, kernel_size=2, stride=2
        )

        # Final occupancy head
        self.occ_conv = F.Conv2d(
            name + ".occ_conv", proj_dim, num_occ_classes, 1, stride=1, padding=0
        )

        # Pre-register inline ops used in __call__
        bs = self.bs
        nf = n_future + 1
        # Compute actual down-sampled spatial size after 2x stride-2 conv
        # Each stride-2 conv: out = floor((in-1)/2) + 1
        Hds = (bev_h - 1) // 2 + 1  # first stride-2
        Hds = (Hds - 1) // 2 + 1  # second stride-2
        Wds = (bev_w - 1) // 2 + 1
        Wds = (Wds - 1) // 2 + 1
        self._Hds = Hds
        self._Wds = Wds
        D = proj_dim
        self._bev_2d_rs = F.Reshape(name + ".bev_2d")
        self._bev_flat_rs = F.Reshape(name + ".bev_flat")
        self._bev_t = F.Transpose(name + ".bev_t", perm=[0, 2, 1])
        self._k_t = F.Transpose(name + ".k_t", perm=[0, 2, 1])
        self._fut_map_rs = F.Reshape(name + ".fut_map")
        self._up_spatial = F.Resize(name + ".up_spatial", scale_factor=float(Hds))
        self._ln_add = F.Add(name + ".ln_add")
        self._qk_mm = F.MatMul(name + ".qk_mm")
        self._sc_mul = F.Mul(name + ".sc_mul")
        self._av_mm = F.MatMul(name + ".av_mm")
        # Pre-register shape constants
        self._bev_2d_shape = F._from_data(
            name + ".bev_2d_shape",
            np.array([bs, embed_dims, bev_h, bev_w], dtype=np.int64),
            is_const=True,
        )
        self._bev_flat_shape = F._from_data(
            name + ".bev_flat_shape",
            np.array([bs, D, Hds * Wds], dtype=np.int64),
            is_const=True,
        )
        self._fut_map_shape = F._from_data(
            name + ".fut_map_shape",
            np.array([bs * nf, D, 1, 1], dtype=np.int64),
            is_const=True,
        )
        self._scale = F._from_data(
            name + ".scale", np.float32(1.0 / (D**0.5)), is_const=True
        )

        super().link_op2module()

        # Pre-register future index tensor
        nf = n_future + 1
        self._fut_idx = F._from_shape(
            name + ".fut_idx", [self.bs, nf], np_dtype=np.int64
        )
        self._tensors[self._fut_idx.name] = self._fut_idx

    def __call__(self, bev_feat, motion_out: dict | None = None):
        """
        bev_feat  : [bs, bev_h*bev_w, embed_dims]
        motion_out: optional dict from MotionHead
        Returns dict with 'occ_pred' [bs, n_future+1, num_occ_classes, bev_h, bev_w]
        """
        bs = self.bs
        bev_h = self.bev_h
        bev_w = self.bev_w
        D = self.proj_dim

        # Reshape BEV to 2D: [bs, E, bev_h, bev_w]
        bev_2d = self._bev_2d_rs(bev_feat, self._bev_2d_shape)

        # Project BEV
        bev_proj = self.bev_proj(bev_2d)  # [bs, D, bev_h, bev_w]

        # Down-scale
        bev_ds = self.ds1(bev_proj)  # [bs, D, ceil(bev_h/2), ceil(bev_w/2)]
        bev_ds = self.ds2(bev_ds)  # [bs, D, Hds, Wds]

        Hds = self._Hds
        Wds = self._Wds

        # Flatten BEV for cross-attention: [bs, D, H, W] -> [bs, H*W, D]
        bev_flat = self._bev_flat_rs(bev_ds, self._bev_flat_shape)
        bev_flat = self._bev_t(bev_flat)  # [bs, Hds*Wds, D]

        # Future query lookup: [bs, n_future+1, D]
        nf = self.n_future + 1
        fut_q = self.future_emb(self._fut_idx)

        # Cross-attention: future query attends to spatial BEV tokens
        # Q: [bs, nf, D], K: [bs, D, Hds*Wds], V: [bs, Hds*Wds, D]
        Q = self.wq(fut_q)  # [bs, nf, D]
        K = self._k_t(self.wk(bev_flat))  # [bs, D, Hds*Wds]
        V = self.wv(bev_flat)  # [bs, Hds*Wds, D]
        # attn: [bs, nf, Hds*Wds]; output: weighted sum over spatial tokens -> [bs, nf, D]
        attn = self.sm(
            self._sc_mul(self._qk_mm(Q, K), self._scale)
        )  # [bs, nf, Hds*Wds]
        out = self.wo(self._av_mm(attn, V))  # [bs, nf, D]
        fut_q = self.ln(self._ln_add(fut_q, out))  # [bs, nf, D]

        # Reshape [bs, nf, D] -> [bs*nf, D, 1, 1] then upsample to [bs*nf, D, Hds, Wds]
        fut_map = self._fut_map_rs(fut_q, self._fut_map_shape)
        # Upsample from 1x1 -> Hds x Wds -> bev_h x bev_w using ConvTranspose2d
        fut_map = self._up_spatial(fut_map)
        # Now [bs*nf, D, Hds, Wds]; upsample 4x to [bs*nf, D, bev_h, bev_w]
        fut_map = self.up1(fut_map)  # [bs*nf, D, Hds*2, Wds*2]
        fut_map = self.up2(fut_map)  # [bs*nf, D, bev_h, bev_w]

        # Predict occupancy
        occ_logits = self.occ_conv(fut_map)  # [bs*nf, num_occ_classes, bev_h, bev_w]

        return {
            "occ_pred": occ_logits,
        }
