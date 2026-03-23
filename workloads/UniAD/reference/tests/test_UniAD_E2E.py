#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end integration tests for the UniAD_E2E ttsim workload.

All tests use very small dimensions to keep runtime fast.
"""

import pytest
import numpy as np
from workloads.UniAD.UniAD_E2E import UniAD_E2E

# ── minimal config ─────────────────────────────────────────────────────────────
_SMALL_CFG = dict(
    embed_dims=32,
    num_query=10,
    num_classes=4,
    bev_h=4,
    bev_w=4,
    num_cameras=2,
    resnet_depth=50,
    num_enc_layers=1,
    num_heads=4,
    ffn_dim=64,
    img_height=32,
    img_width=32,
    bs=1,
    # head-specific overrides
    num_dec_layers=1,
    predict_steps=2,
    num_anchor=2,
    n_future=1,
    planning_steps=2,
    bev_proj_dim=8,
    num_occ_classes=2,
    num_things_classes=2,
    num_stuff_classes=2,
    num_dec_things=1,
    num_dec_stuff=1,
    num_query_things=4,
    num_query_stuff=2,
)


@pytest.mark.unit
def test_e2e_construction():
    """UniAD_E2E should construct without error."""
    model = UniAD_E2E("uniad", _SMALL_CFG)
    assert model is not None


@pytest.mark.unit
def test_e2e_create_input_tensors():
    """create_input_tensors should populate imgs and prev_bev."""
    model = UniAD_E2E("uniad", _SMALL_CFG)
    model.create_input_tensors()
    assert "imgs" in model.input_tensors
    assert "prev_bev" in model.input_tensors


@pytest.mark.unit
def test_e2e_input_shapes():
    """Input tensor shapes should match config."""
    cfg = _SMALL_CFG
    model = UniAD_E2E("uniad", cfg)
    model.create_input_tensors()

    imgs_shape = model.input_tensors["imgs"].shape
    bev_shape = model.input_tensors["prev_bev"].shape

    assert imgs_shape == [
        cfg["bs"],
        cfg["num_cameras"],
        3,
        cfg["img_height"],
        cfg["img_width"],
    ]
    assert bev_shape == [cfg["bs"], cfg["bev_h"] * cfg["bev_w"], cfg["embed_dims"]]


@pytest.mark.unit
def test_e2e_forward():
    """Full forward pass should return plan_traj."""
    model = UniAD_E2E("uniad", _SMALL_CFG)
    model.create_input_tensors()
    out = model()
    assert "plan_traj" in out, f"missing plan_traj key, got: {list(out.keys())}"


@pytest.mark.unit
def test_e2e_plan_traj_shape():
    """plan_traj should have shape [bs, planning_steps, 2]."""
    cfg = _SMALL_CFG
    model = UniAD_E2E("uniad", cfg)
    model.create_input_tensors()
    out = model()
    expected = [cfg["bs"], cfg["planning_steps"], 2]
    assert (
        out["plan_traj"].shape == expected
    ), f"plan_traj shape {out['plan_traj'].shape} != expected {expected}"


@pytest.mark.unit
def test_e2e_graph_non_empty():
    """The forward graph should have ops."""
    model = UniAD_E2E("uniad", _SMALL_CFG)
    model.create_input_tensors()
    _ = model()
    gg = model.get_forward_graph()
    assert gg.get_node_count() > 0, "forward graph is empty"


@pytest.mark.slow
def test_e2e_full_size():
    """Full-size forward pass (marked slow, skipped in fast test suite)."""
    cfg = dict(
        embed_dims=256,
        num_query=900,
        num_classes=10,
        bev_h=50,
        bev_w=50,
        num_cameras=6,
        resnet_depth=50,
        num_enc_layers=6,
        num_heads=8,
        ffn_dim=512,
        img_height=256,
        img_width=256,
        bs=1,
        num_dec_layers=6,
        predict_steps=12,
        num_anchor=6,
        n_future=4,
        planning_steps=6,
        bev_proj_dim=64,
        num_occ_classes=2,
        num_things_classes=8,
        num_stuff_classes=2,
        num_dec_things=6,
        num_dec_stuff=6,
        num_query_things=300,
        num_query_stuff=100,
    )
    model = UniAD_E2E("uniad_full", cfg)
    model.create_input_tensors()
    out = model()
    assert "plan_traj" in out
    assert out["plan_traj"].shape == [1, cfg["planning_steps"], 2]
