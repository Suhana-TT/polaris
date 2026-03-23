#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD BEV encoder modules."""

import pytest
import numpy as np
import ttsim.front.functional.op as F
from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.transformer import (
    BEVFormerEncoder,
)

_SMALL_CFG = dict(
    embed_dims=32,
    num_heads=4,
    ffn_dim=64,
    num_enc_layers=2,
    bev_h=4,
    bev_w=4,
    num_cameras=2,
    num_levels=1,
    bs=1,
    cam_feat_h=4,
    cam_feat_w=4,
)


@pytest.mark.unit
def test_bev_encoder_construction():
    """BEVFormerEncoder should construct without error."""
    enc = BEVFormerEncoder("enc", **_SMALL_CFG)
    assert enc is not None


@pytest.mark.unit
def test_bev_encoder_output_shape():
    """BEV encoder output should be [bs, bev_h*bev_w, embed_dims]."""
    cfg = _SMALL_CFG
    enc = BEVFormerEncoder("enc", **cfg)

    bs = cfg["bs"]
    bev_h = cfg["bev_h"]
    bev_w = cfg["bev_w"]
    E = cfg["embed_dims"]
    nc = cfg["num_cameras"]

    # Dummy multi-level feature maps [bs*nc, C, H, W]
    H, W = 4, 4
    feat = F._from_shape("feat0", [bs * nc, E, H, W])
    mlvl_feats = [feat]

    bev_out = enc(mlvl_feats, prev_bev=None)
    assert bev_out.shape == [
        bs,
        bev_h * bev_w,
        E,
    ], f"unexpected BEV output shape: {bev_out.shape}"


@pytest.mark.unit
def test_bev_encoder_with_prev_bev():
    """BEV encoder should accept an explicit prev_bev tensor."""
    cfg = _SMALL_CFG
    enc = BEVFormerEncoder("enc2", **cfg)

    bs = cfg["bs"]
    bev_h = cfg["bev_h"]
    bev_w = cfg["bev_w"]
    E = cfg["embed_dims"]
    nc = cfg["num_cameras"]

    feat = F._from_shape("feat0b", [bs * nc, E, 4, 4])
    prev_bev = F._from_shape("prev_bev", [bs, bev_h * bev_w, E])

    bev_out = enc([feat], prev_bev=prev_bev)
    assert bev_out.shape[0] == bs
    assert bev_out.shape[1] == bev_h * bev_w
    assert bev_out.shape[2] == E
