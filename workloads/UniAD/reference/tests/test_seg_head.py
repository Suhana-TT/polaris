#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD PansegformerHead (segmentation head)."""

import pytest
import numpy as np
import ttsim.front.functional.op as F
from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.panseg_head import (
    PansegformerHead,
)

_SMALL_CFG = dict(
    embed_dims=32,
    bev_h=4,
    bev_w=4,
    num_heads=4,
    num_things_classes=4,
    num_stuff_classes=2,
    num_dec_things=2,
    num_dec_stuff=2,
    num_query_things=8,
    num_query_stuff=4,
    canvas_h=16,
    canvas_w=16,
    bs=1,
)


@pytest.mark.unit
def test_seg_head_construction():
    head = PansegformerHead("seg", _SMALL_CFG)
    assert head is not None


@pytest.mark.unit
def test_seg_head_output_keys():
    cfg = _SMALL_CFG
    head = PansegformerHead("seg", cfg)
    E = cfg["embed_dims"]
    bev = F._from_shape("bev", [cfg["bs"], cfg["bev_h"] * cfg["bev_w"], E])
    out = head(bev)
    for key in ("thing_cls", "thing_masks", "stuff_cls", "stuff_masks"):
        assert key in out, f"missing key: {key}"


@pytest.mark.unit
def test_seg_head_thing_cls_shape():
    cfg = _SMALL_CFG
    head = PansegformerHead("seg2", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    bev = F._from_shape("bev2", [bs, cfg["bev_h"] * cfg["bev_w"], E])
    out = head(bev)
    assert out["thing_cls"].shape == [
        bs,
        cfg["num_query_things"],
        cfg["num_things_classes"],
    ]


@pytest.mark.unit
def test_seg_head_stuff_cls_shape():
    cfg = _SMALL_CFG
    head = PansegformerHead("seg3", cfg)
    E = cfg["embed_dims"]
    bs = cfg["bs"]
    bev = F._from_shape("bev3", [bs, cfg["bev_h"] * cfg["bev_w"], E])
    out = head(bev)
    assert out["stuff_cls"].shape == [
        bs,
        cfg["num_query_stuff"],
        cfg["num_stuff_classes"],
    ]
