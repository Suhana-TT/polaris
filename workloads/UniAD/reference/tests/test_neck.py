#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD FPN neck ttsim implementation."""

import pytest
import numpy as np
import ttsim.front.functional.op as F
from workloads.UniAD.projects.mmdet3d_plugin.uniad.detectors.uniad_track import FPNNeck


def _make_dummy_features(bs=1, spatial=8):
    """Create 4 dummy feature tensors as would come from the backbone."""
    channels = [256, 512, 1024, 2048]
    sizes = [spatial, spatial // 2, spatial // 4, spatial // 8]
    feats = []
    for c, h in zip(channels, sizes):
        t = F._from_shape(f"feat_c{c}", [bs, c, max(h, 1), max(h, 1)])
        feats.append(t)
    return feats


@pytest.mark.unit
def test_fpn_construction():
    """FPN neck should construct without error."""
    neck = FPNNeck("neck", in_channels=[256, 512, 1024, 2048], out_channels=256)
    assert neck is not None


@pytest.mark.unit
def test_fpn_output_count():
    """FPN should return the same number of levels as input."""
    neck = FPNNeck("neck", in_channels=[256, 512, 1024, 2048], out_channels=256)
    feats = _make_dummy_features()
    outs = neck(feats)
    assert len(outs) == 4, f"expected 4 FPN outputs, got {len(outs)}"


@pytest.mark.unit
def test_fpn_output_channels():
    """All FPN outputs should have out_channels channels."""
    out_ch = 128
    neck = FPNNeck("neck2", in_channels=[256, 512, 1024, 2048], out_channels=out_ch)
    feats = _make_dummy_features()
    outs = neck(feats)
    for i, p in enumerate(outs):
        assert (
            p.shape[1] == out_ch
        ), f"P{i+2} channel mismatch: got {p.shape[1]}, expected {out_ch}"


@pytest.mark.unit
def test_fpn_custom_levels():
    """FPN should work with custom number of levels."""
    neck = FPNNeck("neck3", in_channels=[64, 128], out_channels=64)
    feats = [
        F._from_shape("f0", [1, 64, 16, 16]),
        F._from_shape("f1", [1, 128, 8, 8]),
    ]
    outs = neck(feats)
    assert len(outs) == 2
    for p in outs:
        assert p.shape[1] == 64
