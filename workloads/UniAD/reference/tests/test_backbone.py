#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for UniAD ResNet backbone ttsim implementation."""

import pytest
import numpy as np
from workloads.UniAD.projects.mmdet3d_plugin.uniad.detectors.uniad_track import (
    ResNetBackbone,
    Bottleneck,
)

# ─── small config to keep tests fast ──────────────────────────────────────────
_SMALL_CFG = {
    "bs": 1,
    "num_channels": 3,
    "img_height": 32,
    "img_width": 32,
    "resnet_layers": [1, 1, 1, 1],  # minimal depth
}


@pytest.mark.unit
def test_bottleneck_no_downsample():
    """Bottleneck without down-sample: output shape should equal input shape."""
    blk = Bottleneck(
        "bb",
        in_channels=256,
        mid_channels=64,
        out_channels=256,
        stride=1,
        downsample=False,
    )
    x = blk.create_shape_tensor("x", [1, 256, 8, 8])
    y = blk(x)
    assert y.shape == [1, 256, 8, 8], f"unexpected shape {y.shape}"


@pytest.mark.unit
def test_bottleneck_with_downsample():
    """Bottleneck with stride-2 down-sample."""
    blk = Bottleneck(
        "bb_ds",
        in_channels=64,
        mid_channels=64,
        out_channels=256,
        stride=2,
        downsample=True,
    )
    x = blk.create_shape_tensor("x", [1, 64, 16, 16])
    y = blk(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 256
    assert y.shape[2] == 8  # H / 2
    assert y.shape[3] == 8


@pytest.mark.unit
def test_backbone_construction():
    """Backbone should be constructible without error."""
    model = ResNetBackbone("backbone", _SMALL_CFG)
    model.create_input_tensors()


@pytest.mark.unit
def test_backbone_output_shapes():
    """Backbone should produce 4 multi-scale feature maps."""
    model = ResNetBackbone("backbone", _SMALL_CFG)
    model.create_input_tensors()
    outs = model()
    assert len(outs) == 4, f"expected 4 outputs, got {len(outs)}"
    # channels should be 256, 512, 1024, 2048
    expected_channels = [256, 512, 1024, 2048]
    for i, (out, ec) in enumerate(zip(outs, expected_channels)):
        assert (
            out.shape[1] == ec
        ), f"C{i+2} channel mismatch: got {out.shape[1]}, expected {ec}"


@pytest.mark.unit
def test_backbone_graph():
    """Backbone graph should be non-empty and buildable."""
    model = ResNetBackbone("backbone", _SMALL_CFG)
    model.create_input_tensors()
    _ = model()
    gg = model.get_forward_graph()
    assert gg.get_node_count() > 0, "forward graph is empty"


@pytest.mark.unit
def test_backbone_spatial_downsampling():
    """Each stage should reduce spatial resolution."""
    model = ResNetBackbone("backbone", _SMALL_CFG)
    model.create_input_tensors()
    outs = model()
    # spatial sizes should be strictly decreasing from c2 -> c5
    prev_spatial = outs[0].shape[2] * outs[0].shape[3]
    for i in range(1, 4):
        curr_spatial = outs[i].shape[2] * outs[i].shape[3]
        assert curr_spatial <= prev_spatial, (
            f"C{i+3} not smaller than C{i+2}: " f"{curr_spatial} vs {prev_spatial}"
        )
        prev_spatial = curr_spatial
