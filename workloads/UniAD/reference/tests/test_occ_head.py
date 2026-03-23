#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD OccHead (occupancy prediction head)."""

import pytest
import numpy as np
import ttsim.front.functional.op as F
from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head import OccHead

_SMALL_CFG = dict(
    embed_dims=32,
    bev_h=8,
    bev_w=8,
    n_future=2,
    bev_proj_dim=16,
    num_occ_classes=2,
    bs=1,
)


@pytest.mark.unit
def test_occ_head_construction():
    head = OccHead("occ", _SMALL_CFG)
    assert head is not None


@pytest.mark.unit
def test_occ_head_output_key():
    cfg = _SMALL_CFG
    head = OccHead("occ", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    bev_h = cfg["bev_h"]
    bev_w = cfg["bev_w"]
    bev = F._from_shape("bev", [bs, bev_h * bev_w, E])
    out = head(bev)
    assert "occ_pred" in out, f"missing key: occ_pred"


@pytest.mark.unit
def test_occ_head_output_shape():
    """occ_pred should be [bs*(n_future+1), num_occ_classes, bev_h, bev_w]."""
    cfg = _SMALL_CFG
    head = OccHead("occ2", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    bev_h = cfg["bev_h"]
    bev_w = cfg["bev_w"]
    nf = cfg["n_future"]
    nc = cfg["num_occ_classes"]
    bev = F._from_shape("bev2", [bs, bev_h * bev_w, E])
    out = head(bev)
    pred = out["occ_pred"]
    assert pred.shape[0] == bs * (nf + 1)
    assert pred.shape[1] == nc
    assert pred.shape[2] == bev_h
    assert pred.shape[3] == bev_w
