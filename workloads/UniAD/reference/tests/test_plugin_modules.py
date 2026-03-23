#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Tests for UniAD plugin and utility modules.

Covers:
  - track_head_plugin: MemoryBank, QueryInteractionModule, RuntimeTrackerBase, Instances
  - occ_head_plugin: BevFeatureSlicer, MLP, SimpleConv2d, CVT_Decoder, UpsamplingAdd, Bottleneck
  - motion_head_plugin: IntentionInteraction, TrackAgentInteraction, MapInteraction,
                        MotionDeformableAttention, CustomModeMultiheadAttention
  - modules: MultiScaleDeformableAttnFunction_fp32, CustomMSDeformableAttention,
             inverse_sigmoid, get_reference_points
  - utility: calculate_birds_eye_view_parameters
"""

import pytest
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# ─── track_head_plugin ────────────────────────────────────────────────────────


class TestMemoryBank:
    @pytest.mark.unit
    def test_construction(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.modules import (
            MemoryBank,
        )

        args = {"memory_bank_score_thresh": 0.5, "memory_bank_len": 4}
        mb = MemoryBank(args, dim_in=32, hidden_dim=64, dim_out=32)
        assert isinstance(mb, SimNN.Module)

    @pytest.mark.unit
    def test_noop_forward(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.modules import (
            MemoryBank,
        )
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import (
            Instances,
        )

        args = {"memory_bank_score_thresh": 0.5, "memory_bank_len": 4}
        mb = MemoryBank(args, dim_in=32, hidden_dim=64, dim_out=32)
        ti = Instances((200, 200))
        ti.scores = np.array([0.9, 0.3])
        result = mb(ti)
        assert result is ti  # no-op returns same object


class TestQueryInteractionModule:
    @pytest.mark.unit
    def test_construction(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.modules import (
            QueryInteractionModule,
        )

        args = {
            "random_drop": 0.0,
            "fp_ratio": 0.0,
            "update_query_pos": False,
            "merger_dropout": 0.0,
        }
        qim = QueryInteractionModule(args, dim_in=32, hidden_dim=64, dim_out=32)
        assert isinstance(qim, SimNN.Module)

    @pytest.mark.unit
    def test_merged_output(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.modules import (
            QueryInteractionModule,
        )
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import (
            Instances,
        )

        args = {
            "random_drop": 0.0,
            "fp_ratio": 0.0,
            "update_query_pos": False,
            "merger_dropout": 0.0,
        }
        qim = QueryInteractionModule(args, dim_in=32, hidden_dim=64, dim_out=32)

        init_ti = Instances((200, 200))
        init_ti.obj_idxes = np.array([-1, -1])
        init_ti.scores = np.array([0.0, 0.0])

        active_ti = Instances((200, 200))
        active_ti.obj_idxes = np.array([0, 1])
        active_ti.scores = np.array([0.9, 0.8])

        data = {"track_instances": active_ti, "init_track_instances": init_ti}
        merged = qim(data)
        assert len(merged) == 4  # 2 init + 2 active


class TestRuntimeTrackerBase:
    @pytest.mark.unit
    def test_assigns_new_tracks(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.tracker import (
            RuntimeTrackerBase,
        )
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import (
            Instances,
        )

        tracker = RuntimeTrackerBase(
            score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5
        )
        ti = Instances((200, 200))
        ti.obj_idxes = np.array([-1, -1, -1])
        ti.scores = np.array([0.9, 0.3, 0.8])
        ti.disappear_time = np.array([0, 0, 0])
        tracker.update(ti)
        # Tracks with score >= 0.5 get assigned IDs
        assert ti.obj_idxes[0] >= 0
        assert ti.obj_idxes[2] >= 0
        assert ti.obj_idxes[1] == -1  # low-score track stays unassigned

    @pytest.mark.unit
    def test_removes_old_tracks(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.tracker import (
            RuntimeTrackerBase,
        )
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import (
            Instances,
        )

        tracker = RuntimeTrackerBase(
            score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=2
        )
        ti = Instances((200, 200))
        ti.obj_idxes = np.array([5])  # already-assigned track
        ti.scores = np.array([0.1])  # below filter threshold
        ti.disappear_time = np.array([2])  # at miss_tolerance
        tracker.update(ti)
        assert ti.obj_idxes[0] == -1  # killed


class TestInstances:
    @pytest.mark.unit
    def test_len_and_getitem(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import (
            Instances,
        )

        inst = Instances((200, 200))
        inst.scores = np.array([0.9, 0.5, 0.1])
        assert len(inst) == 3
        sub = inst[0:2]
        assert len(sub) == 2

    @pytest.mark.unit
    def test_cat(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin.track_instance import (
            Instances,
        )

        a = Instances((200, 200))
        a.scores = np.array([0.9, 0.8])
        b = Instances((200, 200))
        b.scores = np.array([0.3])
        merged = Instances.cat([a, b])
        assert len(merged) == 3
        np.testing.assert_allclose(merged.scores, [0.9, 0.8, 0.3])


# ─── occ_head_plugin ──────────────────────────────────────────────────────────


class TestBevFeatureSlicer:
    @pytest.mark.unit
    def test_identity_mapping(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            BevFeatureSlicer,
        )

        grid_conf = {
            "xbound": [-50, 50, 0.5],
            "ybound": [-50, 50, 0.5],
            "zbound": [-10, 10, 20.0],
        }
        slicer = BevFeatureSlicer(grid_conf, grid_conf)
        assert slicer.identity_mapping is True
        x = F._from_shape("bev_x", [1, 32, 200, 200])
        out = slicer(x)
        assert out is x

    @pytest.mark.unit
    def test_non_identity_construction(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            BevFeatureSlicer,
        )

        gc = {
            "xbound": [-50, 50, 0.5],
            "ybound": [-50, 50, 0.5],
            "zbound": [-10, 10, 20.0],
        }
        mgc = {
            "xbound": [-30, 30, 1.0],
            "ybound": [-30, 30, 1.0],
            "zbound": [-10, 10, 20.0],
        }
        slicer = BevFeatureSlicer(gc, mgc)
        assert slicer.identity_mapping is False


class TestMLP:
    @pytest.mark.unit
    def test_construction_and_forward(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            MLP,
        )

        mlp = MLP("test_mlp", input_dim=32, hidden_dim=64, output_dim=16, num_layers=3)
        assert isinstance(mlp, SimNN.Module)
        x = F._from_shape("mlp_in", [1, 10, 32])
        out = mlp(x)
        assert out.shape[-1] == 16


class TestSimpleConv2d:
    @pytest.mark.unit
    def test_single_conv(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            SimpleConv2d,
        )

        conv = SimpleConv2d(in_channels=32, out_channels=64, num_conv=1)
        assert isinstance(conv, SimNN.Module)

    @pytest.mark.unit
    def test_multi_conv(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            SimpleConv2d,
        )

        conv = SimpleConv2d(
            in_channels=32, out_channels=64, conv_channels=48, num_conv=3
        )
        assert isinstance(conv, SimNN.Module)


class TestCVTDecoder:
    @pytest.mark.unit
    def test_construction(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            CVT_Decoder,
        )

        decoder = CVT_Decoder(dim=32, blocks=[64, 128, 64])
        assert isinstance(decoder, SimNN.Module)

    @pytest.mark.unit
    def test_multi_block_construction(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            CVT_Decoder,
        )

        decoder = CVT_Decoder(dim=32, blocks=[64, 128])
        assert len(decoder._layers_list) == 2


class TestUpsamplingAdd:
    @pytest.mark.unit
    def test_construction(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            UpsamplingAdd,
        )

        ua = UpsamplingAdd("test_ua", in_channels=32, out_channels=64)
        assert isinstance(ua, SimNN.Module)


class TestBottleneck:
    @pytest.mark.unit
    def test_construction_plain(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            Bottleneck,
        )

        bn = Bottleneck(in_channels=64, out_channels=64)
        assert isinstance(bn, SimNN.Module)

    @pytest.mark.unit
    def test_construction_with_proj(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.modules import (
            Bottleneck,
        )

        bn = Bottleneck(in_channels=64, out_channels=128)
        assert bn.has_proj is True


# ─── motion_head_plugin ───────────────────────────────────────────────────────

_motion_available = pytest.importorskip.__module__  # noqa: keep import


def _import_motion(name):
    """Import from motion_head_plugin, skipping if casadi is missing."""
    casadi = pytest.importorskip("casadi", reason="casadi not installed")  # noqa: F841
    import importlib

    return importlib.import_module(
        f"workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.{name}"
    )


class TestIntentionInteraction:
    @pytest.mark.unit
    def test_construction(self):
        mods = _import_motion("modules")
        ii = mods.IntentionInteraction(embed_dims=32, num_heads=4)
        assert isinstance(ii, SimNN.Module)

    @pytest.mark.unit
    def test_forward(self):
        mods = _import_motion("modules")
        ii = mods.IntentionInteraction(embed_dims=32, num_heads=4)
        q = F._from_shape("ii_q", [1, 5, 3, 32])
        out = ii(q)
        assert out.shape == [1, 5, 3, 32]


class TestTrackAgentInteraction:
    @pytest.mark.unit
    def test_construction(self):
        mods = _import_motion("modules")
        tai = mods.TrackAgentInteraction(embed_dims=32, num_heads=4)
        assert isinstance(tai, SimNN.Module)

    @pytest.mark.unit
    def test_forward(self):
        mods = _import_motion("modules")
        tai = mods.TrackAgentInteraction(embed_dims=32, num_heads=4)
        q = F._from_shape("tai_q", [1, 5, 3, 32])
        k = F._from_shape("tai_k", [1, 5, 32])
        out = tai(q, k)
        assert out.shape == [1, 5, 3, 32]


class TestMapInteraction:
    @pytest.mark.unit
    def test_construction(self):
        mods = _import_motion("modules")
        mi = mods.MapInteraction(embed_dims=32, num_heads=4)
        assert isinstance(mi, SimNN.Module)

    @pytest.mark.unit
    def test_forward(self):
        mods = _import_motion("modules")
        mi = mods.MapInteraction(embed_dims=32, num_heads=4)
        q = F._from_shape("mi_q", [1, 5, 3, 32])
        k = F._from_shape("mi_k", [1, 10, 32])
        out = mi(q, k)
        assert out.shape == [1, 5, 3, 32]


class TestMotionDeformableAttention:
    @pytest.mark.unit
    def test_construction(self):
        mda = _import_motion("motion_deformable_attn")
        attn = mda.MotionDeformableAttention(
            embed_dims=32, num_heads=4, num_levels=2, num_points=2
        )
        assert isinstance(attn, SimNN.Module)

    @pytest.mark.unit
    def test_has_expected_projections(self):
        mda = _import_motion("motion_deformable_attn")
        attn = mda.MotionDeformableAttention(
            embed_dims=32, num_heads=4, num_levels=2, num_points=2
        )
        assert hasattr(attn, "sampling_offsets")
        assert hasattr(attn, "attention_weights")
        assert hasattr(attn, "value_proj")


class TestCustomModeMultiheadAttention:
    @pytest.mark.unit
    def test_construction(self):
        mda = _import_motion("motion_deformable_attn")
        attn = mda.CustomModeMultiheadAttention(embed_dims=32, num_heads=4)
        assert isinstance(attn, SimNN.Module)

    @pytest.mark.unit
    def test_forward(self):
        mda = _import_motion("motion_deformable_attn")
        attn = mda.CustomModeMultiheadAttention(embed_dims=32, num_heads=4)
        q = F._from_shape("cmma_q", [1, 10, 32])
        out = attn(q)
        assert out.shape == [1, 10, 32]


# ─── modules/multi_scale_deformable_attn_function ─────────────────────────────


class TestMultiScaleDeformableAttnFunction:
    @pytest.mark.unit
    def test_function_importable(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.multi_scale_deformable_attn_function import (
            multi_scale_deformable_attn_ttsim,
            MultiScaleDeformableAttnFunction_fp32,
            MultiScaleDeformableAttnFunction_fp16,
        )

        assert callable(multi_scale_deformable_attn_ttsim)

    @pytest.mark.unit
    def test_fp16_equals_fp32(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.multi_scale_deformable_attn_function import (
            MultiScaleDeformableAttnFunction_fp32,
            MultiScaleDeformableAttnFunction_fp16,
        )

        assert (
            MultiScaleDeformableAttnFunction_fp16
            is MultiScaleDeformableAttnFunction_fp32
        )

    @pytest.mark.unit
    def test_apply_runs(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.multi_scale_deformable_attn_function import (
            MultiScaleDeformableAttnFunction_fp32,
        )

        bs, nq, nH, dH = 1, 4, 2, 8
        nL, nP = 1, 2
        value = F._from_shape("msda_v", [bs, 4, nH, dH])
        locs = F._from_shape("msda_locs", [bs, nq, nH, nL, nP, 2])
        weights = F._from_shape("msda_w", [bs, nq, nH, nL, nP])
        spatial_shapes = [(2, 2)]
        out = MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes, None, locs, weights, 64
        )
        assert out is not None


# ─── modules/decoder ─────────────────────────────────────────────────────────


class TestCustomMSDeformableAttention:
    @pytest.mark.unit
    def test_construction(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.decoder import (
            CustomMSDeformableAttention,
        )

        attn = CustomMSDeformableAttention(
            name="test_cmsda", embed_dims=32, num_heads=4, num_levels=2, num_points=2
        )
        assert isinstance(attn, SimNN.Module)


class TestInverseSigmoid:
    @pytest.mark.unit
    def test_numerical_correctness(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.decoder import (
            inverse_sigmoid,
        )

        x = np.array([0.5, 0.9, 0.1])
        out = inverse_sigmoid(x)
        # inverse_sigmoid(0.5) == 0.0
        assert abs(out[0]) < 1e-6
        # inverse_sigmoid(0.9) > 0
        assert out[1] > 0.0
        # inverse_sigmoid(0.1) < 0
        assert out[2] < 0.0


# ─── modules/encoder ─────────────────────────────────────────────────────────


class TestGetReferencePoints:
    @pytest.mark.unit
    def test_2d_shape(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.encoder import (
            get_reference_points,
        )

        H, W, bs = 4, 5, 2
        ref = get_reference_points(H, W, dim="2d", bs=bs)
        assert ref.shape == (bs, H * W, 1, 2)
        assert (ref >= 0).all() and (ref <= 1).all()

    @pytest.mark.unit
    def test_3d_shape(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.modules.encoder import (
            get_reference_points,
        )

        H, W, Z, nP, bs = 3, 4, 8, 4, 1
        ref = get_reference_points(H, W, Z=Z, num_points_in_pillar=nP, dim="3d", bs=bs)
        assert ref.shape == (bs, H * W, nP, 3)


# ─── occ_head_plugin/utils ───────────────────────────────────────────────────


class TestCalculateBirdsEyeViewParameters:
    @pytest.mark.unit
    def test_output_values(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.utils import (
            calculate_birds_eye_view_parameters,
        )

        xb = [-50.0, 50.0, 0.5]
        yb = [-50.0, 50.0, 0.5]
        zb = [-10.0, 10.0, 20.0]
        res, start, dim = calculate_birds_eye_view_parameters(xb, yb, zb)
        assert res[0] == pytest.approx(0.5)
        assert dim[0] == 200  # (100 / 0.5)
        assert dim[1] == 200

    @pytest.mark.unit
    def test_output_shapes(self):
        from workloads.UniAD.projects.mmdet3d_plugin.uniad.dense_heads.occ_head_plugin.utils import (
            calculate_birds_eye_view_parameters,
        )

        res, start, dim = calculate_birds_eye_view_parameters(
            [-10, 10, 1.0], [-10, 10, 1.0], [-5, 5, 10.0]
        )
        assert res.shape == (3,)
        assert start.shape == (3,)
        assert dim.shape == (3,)
