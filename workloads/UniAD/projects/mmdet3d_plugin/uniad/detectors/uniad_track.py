#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim ResNet-50 backbone and FPN neck for UniAD.

ResNetBackbone produces multi-scale feature maps [C2, C3, C4, C5] with channels
[256, 512, 1024, 2048] suitable for the FPN neck.

FPNNeck takes multi-scale backbone features [C2, C3, C4, C5] and
produces a list of same-channel feature maps [P2, P3, P4, P5].
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class Bottleneck(SimNN.Module):
    """
    ResNet Bottleneck block: 1x1 -> 3x3 -> 1x1 with optional down-sample.

    in_channels  : channels coming in
    mid_channels : width of the 3x3 conv
    out_channels : output channels (= mid_channels * 4)
    stride       : stride for the 3x3 conv (and down-sample if any)
    downsample   : bool, add a projection shortcut?
    """

    expansion: int = 4

    def __init__(
        self,
        name: str,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
    ):
        super().__init__()
        self.name = name
        self._has_ds = downsample

        self.conv1 = F.Conv2d(
            name + ".conv1", in_channels, mid_channels, 1, stride=1, padding=0
        )
        self.bn1 = F.BatchNorm2d(name + ".bn1", mid_channels)
        self.relu1 = F.Relu(name + ".relu1")

        self.conv2 = F.Conv2d(
            name + ".conv2", mid_channels, mid_channels, 3, stride=stride, padding=1
        )
        self.bn2 = F.BatchNorm2d(name + ".bn2", mid_channels)
        self.relu2 = F.Relu(name + ".relu2")

        self.conv3 = F.Conv2d(
            name + ".conv3", mid_channels, out_channels, 1, stride=1, padding=0
        )
        self.bn3 = F.BatchNorm2d(name + ".bn3", out_channels)
        self.relu3 = F.Relu(name + ".relu3")

        self.add = F.Add(name + ".add")

        if downsample:
            self.conv_ds = F.Conv2d(
                name + ".conv_ds",
                in_channels,
                out_channels,
                1,
                stride=stride,
                padding=0,
            )
            self.bn_ds = F.BatchNorm2d(name + ".bn_ds", out_channels)

        super().link_op2module()

    def __call__(self, x):
        residual = x
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self._has_ds:
            residual = self.bn_ds(self.conv_ds(x))
        return self.relu3(self.add(y, residual))


class ResNetBackbone(SimNN.Module):
    """
    ResNet backbone.

    For ResNet-50 set layers=[3,4,6,3].
    Returns a list [c2, c3, c4, c5] – feature maps from stages 1-4.
    """

    _STAGE_CHANNELS = [64, 128, 256, 512]  # mid-channels per stage
    _STEM_CHANNELS = 64

    def __init__(self, name: str, cfg: dict):
        super().__init__()
        self.name = name

        self.bs = cfg.get("bs", 1)
        self.num_channels = cfg.get("num_channels", 3)
        self.img_height = cfg.get("img_height", 256)
        self.img_width = cfg.get("img_width", 256)
        layers = cfg.get("resnet_layers", [3, 4, 6, 3])

        # ── stem ───────────────────────────────────────────────────────────
        self.stem_conv = F.Conv2d(
            name + ".stem.conv",
            self.num_channels,
            self._STEM_CHANNELS,
            7,
            stride=2,
            padding=3,
        )
        self.stem_bn = F.BatchNorm2d(name + ".stem.bn", self._STEM_CHANNELS)
        self.stem_relu = F.Relu(name + ".stem.relu")
        self.stem_pool = F.MaxPool2d(
            name + ".stem.pool", kernel_size=3, stride=2, padding=1
        )

        # ── stages ─────────────────────────────────────────────────────────
        in_ch = self._STEM_CHANNELS
        self.stage1 = SimNN.ModuleList(
            self._make_stage(name + ".stage1", 0, layers[0], in_ch, 64, stride=1)
        )
        in_ch = 64 * Bottleneck.expansion  # 256

        self.stage2 = SimNN.ModuleList(
            self._make_stage(name + ".stage2", 1, layers[1], in_ch, 128, stride=2)
        )
        in_ch = 128 * Bottleneck.expansion  # 512

        self.stage3 = SimNN.ModuleList(
            self._make_stage(name + ".stage3", 2, layers[2], in_ch, 256, stride=2)
        )
        in_ch = 256 * Bottleneck.expansion  # 1024

        self.stage4 = SimNN.ModuleList(
            self._make_stage(name + ".stage4", 3, layers[3], in_ch, 512, stride=2)
        )

        # cache input tensors (populated in create_input_tensors)
        self.input_tensors: dict = {}

        super().link_op2module()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _make_stage(self, prefix, stage_idx, num_blocks, in_ch, mid_ch, stride):
        out_ch = mid_ch * Bottleneck.expansion
        blocks = []
        # first block – may need down-sample
        downsample = (stride != 1) or (in_ch != out_ch)
        blocks.append(
            Bottleneck(
                f"{prefix}.b0",
                in_ch,
                mid_ch,
                out_ch,
                stride=stride,
                downsample=downsample,
            )
        )
        for i in range(1, num_blocks):
            blocks.append(
                Bottleneck(
                    f"{prefix}.b{i}", out_ch, mid_ch, out_ch, stride=1, downsample=False
                )
            )
        return blocks

    # ── public API ─────────────────────────────────────────────────────────────

    def create_input_tensors(self):
        # imgs: [bs, num_cameras, C, H, W] – flattened across cameras in backbone
        self.input_tensors = {
            "imgs": F._from_shape(
                "imgs",
                [self.bs * 6, self.num_channels, self.img_height, self.img_width],
                is_param=False,
                np_dtype=np.float32,
            )
        }

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def __call__(self, x=None):
        if x is None:
            x = self.input_tensors["imgs"]

        # stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        # stage 1
        for blk in self.stage1:
            x = blk(x)
        c2 = x

        # stage 2
        for blk in self.stage2:
            x = blk(x)
        c3 = x

        # stage 3
        for blk in self.stage3:
            x = blk(x)
        c4 = x

        # stage 4
        for blk in self.stage4:
            x = blk(x)
        c5 = x

        return [c2, c3, c4, c5]


# ─── FPN Neck helpers ──────────────────────────────────────────────────────────


class _Conv1x1Block(SimNN.Module):
    def __init__(self, name, in_ch, out_ch):
        super().__init__()
        self.name = name
        self.conv = F.Conv2d(name + ".conv", in_ch, out_ch, 1, stride=1, padding=0)
        self.bn = F.BatchNorm2d(name + ".bn", out_ch)
        super().link_op2module()

    def __call__(self, x):
        return self.bn(self.conv(x))


class _Conv3x3Block(SimNN.Module):
    def __init__(self, name, in_ch, out_ch):
        super().__init__()
        self.name = name
        self.conv = F.Conv2d(name + ".conv", in_ch, out_ch, 3, stride=1, padding=1)
        self.bn = F.BatchNorm2d(name + ".bn", out_ch)
        super().link_op2module()

    def __call__(self, x):
        return self.bn(self.conv(x))


class FPNNeck(SimNN.Module):
    """
    Feature Pyramid Network neck.

    Args:
        name         : module name
        in_channels  : list of input channel counts, one per backbone level
        out_channels : unified output channel count
        num_cameras  : used only for input tensor creation (multi-camera setup)
    """

    def __init__(
        self,
        name: str,
        in_channels: list[int] | None = None,
        out_channels: int = 256,
        num_cameras: int = 6,
    ):
        super().__init__()
        self.name = name
        self.out_channels = out_channels
        self.num_cameras = num_cameras

        if in_channels is None:
            in_channels = [256, 512, 1024, 2048]
        self.in_channels = in_channels
        num_levels = len(in_channels)

        # ── lateral (1×1) convolutions ─────────────────────────────────────
        self.lat_convs = SimNN.ModuleList(
            [
                _Conv1x1Block(f"{name}.lat{i}", ic, out_channels)
                for i, ic in enumerate(in_channels)
            ]
        )

        # ── output (3×3) convolutions ──────────────────────────────────────
        self.out_convs = SimNN.ModuleList(
            [
                _Conv3x3Block(f"{name}.out{i}", out_channels, out_channels)
                for i in range(num_levels)
            ]
        )

        # ── top-down upsample ops (pre-registered so they get linked) ──────
        # Must use setattr so __setattr__ picks them up as SimOpHandles
        self._num_levels = num_levels
        for i in range(1, num_levels):
            setattr(self, f"up_op_{i}", F.Resize(f"{name}.up{i}", scale_factor=2.0))
            setattr(self, f"add_op_{i}", F.Add(f"{name}.add{i}"))

        super().link_op2module()

    def __call__(self, features: list):
        """
        Args:
            features: [c2, c3, c4, c5] – list of SimTensors from backbone

        Returns:
            [p2, p3, p4, p5] – list of SimTensors, all with self.out_channels
        """
        assert len(features) == len(
            self.in_channels
        ), f"FPN expects {len(self.in_channels)} feature levels, got {len(features)}"

        # lateral projections
        laterals = [self.lat_convs[i](features[i]) for i in range(len(features))]

        # top-down path (fuse from deepest to shallowest)
        for i in range(len(laterals) - 1, 0, -1):
            up = getattr(self, f"up_op_{i}")(laterals[i])
            laterals[i - 1] = getattr(self, f"add_op_{i}")(laterals[i - 1], up)

        # output projections
        outs = [self.out_convs[i](laterals[i]) for i in range(len(laterals))]
        return outs
