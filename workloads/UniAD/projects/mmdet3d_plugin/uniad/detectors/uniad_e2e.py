# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim UniAD end-to-end model.

This file is the primary model class, mirroring the original UniAD detector
structure where the full pipeline lives in detectors/uniad_e2e.py.

Pipeline:
    imgs (N cameras) → ResNet backbone → FPN neck
                     → BEVFormer encoder (with prev_bev)
                     → TrackHead → SegHead → MotionHead → OccHead → PlanningHead
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from .uniad_track import ResNetBackbone, FPNNeck
from ..modules.transformer import BEVFormerEncoder
from ..dense_heads.bevformer_head import BEVFormerTrackHead
from ..dense_heads.panseg_head import PansegformerHead
from ..dense_heads.motion_head import MotionHead
from ..dense_heads.occ_head import OccHead
from ..dense_heads.planning_head import PlanningHead
from ..ttsim_utils import DETECTORS


@DETECTORS.register_module()
class UniAD(SimNN.Module):
    """
    Unified Autonomous Driving (UniAD) end-to-end model.

    Args:
        name: module name
        cfg : dict with model configuration

    Config keys:
        embed_dims      : feature dimension (default 256)
        num_query       : number of object queries (default 900)
        num_classes     : number of detection classes (default 10)
        bev_h, bev_w    : BEV grid height/width (default 50x50)
        num_cameras     : number of cameras (default 6)
        num_enc_layers  : BEV encoder layers (default 6)
        num_heads       : attention heads (default 8)
        ffn_dim         : FFN hidden dim (default 512)
        img_height      : input image height (default 256)
        img_width       : input image width  (default 256)
        bs              : batch size (default 1)
    """

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name

        # ── extract config ────────────────────────────────────────────────
        embed_dims = cfg.get("embed_dims", 256)
        num_cameras = cfg.get("num_cameras", 6)
        num_enc = cfg.get("num_enc_layers", 6)
        num_heads = cfg.get("num_heads", 8)
        ffn_dim = cfg.get("ffn_dim", 512)
        bev_h = cfg.get("bev_h", 50)
        bev_w = cfg.get("bev_w", 50)
        img_h = cfg.get("img_height", 256)
        img_w = cfg.get("img_width", 256)
        bs = cfg.get("bs", 1)

        self.bs = bs
        self.embed_dims = embed_dims
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        self.img_h = img_h
        self.img_w = img_w

        # ── backbone ─────────────────────────────────────────────────────
        bb_cfg = {
            "bs": bs,
            "num_channels": 3,
            "img_height": img_h,
            "img_width": img_w,
            "resnet_layers": [3, 4, 6, 3],  # ResNet-50
        }
        self.backbone = ResNetBackbone(name + ".backbone", bb_cfg)

        # ── neck ──────────────────────────────────────────────────────────
        self.neck = FPNNeck(
            name + ".neck",
            in_channels=[256, 512, 1024, 2048],
            out_channels=embed_dims,
            num_cameras=num_cameras,
        )

        # ── BEV encoder ───────────────────────────────────────────────────
        cam_feat_h = img_h // 4
        cam_feat_w = img_w // 4
        self.bev_encoder = BEVFormerEncoder(
            name + ".bev_encoder",
            embed_dims=embed_dims,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_enc_layers=num_enc,
            bev_h=bev_h,
            bev_w=bev_w,
            num_cameras=num_cameras,
            num_levels=4,
            bs=bs,
            cam_feat_h=cam_feat_h,
            cam_feat_w=cam_feat_w,
        )

        # ── task heads ───────────────────────────────────────────────────
        head_cfg = {**cfg, "num_dec_layers": cfg.get("num_dec_layers", 6)}
        self.track_head = BEVFormerTrackHead(name + ".track_head", head_cfg)
        self.seg_head = PansegformerHead(name + ".seg_head", head_cfg)
        self.motion_head = MotionHead(name + ".motion_head", head_cfg)
        self.occ_head = OccHead(name + ".occ_head", head_cfg)
        self.planning_head = PlanningHead(name + ".plan_head", head_cfg)

        # ── pre-registered reshape: [bs, nc, 3, H, W] → [bs*nc, 3, H, W] ─
        self._imgs_flat_reshape = F.Reshape(name + ".imgs_flat")
        self._imgs_flat_shape = F._from_data(
            name + ".imgs_flat_shape",
            np.array([bs * num_cameras, 3, img_h, img_w], dtype=np.int64),
            is_const=True,
        )

        # input tensors populated by create_input_tensors()
        self.input_tensors: dict = {}

        super().link_op2module()

    # ── input tensor creation ─────────────────────────────────────────────

    def create_input_tensors(self):
        """Create SimTensor inputs for graph recording."""
        imgs = F._from_shape(
            "imgs",
            [self.bs, self.num_cameras, 3, self.img_h, self.img_w],
            is_param=False,
            np_dtype=np.float32,
        )
        prev_bev = F._from_shape(
            "prev_bev",
            [self.bs, self.bev_h * self.bev_w, self.embed_dims],
            is_param=False,
            np_dtype=np.float32,
        )
        self.input_tensors = {"imgs": imgs, "prev_bev": prev_bev}

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    # ── forward pass ─────────────────────────────────────────────────────

    def __call__(self, imgs=None, prev_bev=None):
        if imgs is None:
            imgs = self.input_tensors["imgs"]
        if prev_bev is None:
            prev_bev = self.input_tensors["prev_bev"]

        # [bs, nc, C, H, W] → [bs*nc, C, H, W]
        imgs_flat = self._imgs_flat_reshape(imgs, self._imgs_flat_shape)

        # 1. Backbone
        img_feats = self.backbone(imgs_flat)

        # 2. FPN neck
        fpn_feats = self.neck(img_feats)

        # 3. BEV encoder → [bs, bev_h*bev_w, embed_dims]
        bev_feat = self.bev_encoder(fpn_feats, prev_bev)

        # 4. Task heads
        track_out = self.track_head(bev_feat)
        seg_out = self.seg_head(bev_feat)  # noqa: F841
        motion_out = self.motion_head(bev_feat, track_out)
        occ_out = self.occ_head(bev_feat, motion_out)
        plan_out = self.planning_head(bev_feat, motion_out, occ_out)

        return plan_out

    # ── training stubs ────────────────────────────────────────────────────

    def forward_train(self, **kwargs):
        raise NotImplementedError("UniAD.forward_train: training-only")

    def forward_test(self, **kwargs):
        raise NotImplementedError("UniAD.forward_test: use __call__ for simulation")
