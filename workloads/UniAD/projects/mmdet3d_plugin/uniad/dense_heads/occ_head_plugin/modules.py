# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: occ_head_plugin/modules.py — SimNN replacements.
No torch, no mmcv, no einops imports.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from .utils import calculate_birds_eye_view_parameters


class BevFeatureSlicer(SimNN.Module):
    """TTSim SimNN module for BEV feature slicing via grid sampling."""

    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()
        self.name = "bev_feature_slicer"
        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False
            bev_resolution, bev_start_position, bev_dimension = (
                calculate_birds_eye_view_parameters(
                    grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"]
                )
            )
            map_bev_resolution, map_bev_start_position, map_bev_dimension = (
                calculate_birds_eye_view_parameters(
                    map_grid_conf["xbound"],
                    map_grid_conf["ybound"],
                    map_grid_conf["zbound"],
                )
            )
            self.grid_sample = F.GridSample(
                self.name + ".grid_sample", mode="bilinear", align_corners=True
            )
        super().link_op2module()

    def __call__(self, x):
        if self.identity_mapping:
            return x
        return self.grid_sample(x, x)  # grid computed at runtime


class MLP(SimNN.Module):
    """TTSim SimNN multi-layer perceptron."""

    def __init__(self, name, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.name = name
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self._linears = []
        self._relus = []
        for i in range(num_layers):
            lin = SimNN.Linear(name + f".layer{i}", dims[i], dims[i + 1])
            self._linears.append(lin)
            setattr(self, f"layer{i}", lin)
            if i < num_layers - 1:
                relu = F.Relu(name + f".relu{i}")
                self._relus.append(relu)
                setattr(self, f"relu{i}", relu)
        super().link_op2module()

    def __call__(self, x):
        for i, lin in enumerate(self._linears):
            x = lin(x)
            if i < self.num_layers - 1:
                x = self._relus[i](x)
        return x


class SimpleConv2d(SimNN.Module):
    """TTSim SimNN simple conv2d block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_channels=64,
        num_conv=1,
        conv_cfg=None,
        norm_cfg=None,
        bias="auto",
        init_cfg=None,
    ):
        super().__init__()
        self.name = "simple_conv2d"
        self.out_channels = out_channels
        if num_conv == 1:
            conv_channels = in_channels

        self._convs = []
        self._bns = []
        self._relus = []
        c_in = in_channels
        for i in range(num_conv - 1):
            conv = F.Conv2d(
                self.name + f".conv{i}", c_in, conv_channels, kernel_size=3, padding=1
            )
            bn = F.BatchNorm2d(self.name + f".bn{i}", conv_channels)
            relu = F.Relu(self.name + f".relu{i}")
            self._convs.append(conv)
            self._bns.append(bn)
            self._relus.append(relu)
            setattr(self, f"conv{i}", conv)
            setattr(self, f"bn{i}", bn)
            setattr(self, f"relu{i}", relu)
            c_in = conv_channels
        # Final conv: no norm/relu
        final_conv = F.Conv2d(
            self.name + ".conv_final", c_in, out_channels, kernel_size=1
        )
        self._convs.append(final_conv)
        setattr(self, "conv_final", final_conv)
        super().link_op2module()

    def __call__(self, x):
        for i in range(len(self._relus)):
            x = self._relus[i](self._bns[i](self._convs[i](x)))
        x = self._convs[-1](x)
        return x


class CVT_DecoderBlock(SimNN.Module):
    """TTSim SimNN CVT decoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        skip_dim,
        residual,
        factor,
        upsample,
        with_relu=True,
    ):
        super().__init__()
        self.name = "cvt_decoder_block"
        dim = out_channels // factor
        self.with_relu_flag = with_relu
        self.upsample_flag = upsample
        self.residual_flag = residual

        if upsample:
            self.upsample_op = F.Upsample(
                self.name + ".upsample",
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
        self.conv1 = F.Conv2d(
            self.name + ".conv1", in_channels, dim, kernel_size=3, padding=1
        )
        self.bn1 = F.BatchNorm2d(self.name + ".bn1", dim)
        self.relu1 = F.Relu(self.name + ".relu1")
        self.conv2 = F.Conv2d(self.name + ".conv2", dim, out_channels, kernel_size=1)
        self.bn2 = F.BatchNorm2d(self.name + ".bn2", out_channels)

        if residual:
            self.up_conv = F.Conv2d(
                self.name + ".up_conv", skip_dim, out_channels, kernel_size=1
            )
            self.add = F.Add(self.name + ".add")
        if with_relu:
            self.relu_out = F.Relu(self.name + ".relu_out")
        super().link_op2module()

    def __call__(self, x, skip):
        if self.upsample_flag:
            x = self.upsample_op(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_flag:
            up = self.up_conv(skip)
            x = self.add(x, up)
        if self.with_relu_flag:
            x = self.relu_out(x)
        return x


class CVT_Decoder(SimNN.Module):
    """TTSim SimNN CVT decoder stack."""

    def __init__(
        self,
        dim,
        blocks,
        residual=True,
        factor=2,
        upsample=True,
        use_checkpoint=False,
        init_cfg=None,
    ):
        super().__init__()
        self.name = "cvt_decoder"
        self._layers_list = []
        channels = dim
        for i, out_channels in enumerate(blocks):
            with_relu = i < len(blocks) - 1
            layer = CVT_DecoderBlock(
                channels,
                out_channels,
                dim,
                residual,
                factor,
                upsample,
                with_relu=with_relu,
            )
            layer.name = f"cvt_decoder.block{i}"
            self._layers_list.append(layer)
            setattr(self, f"block{i}", layer)
            channels = out_channels
        self.out_channels = channels
        super().link_op2module()

    def __call__(self, x):
        y = x
        for layer in self._layers_list:
            y = layer(y, x)
        return y


class UpsamplingAdd(SimNN.Module):
    """TTSim SimNN upsampling + add module."""

    def __init__(self, name, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.name = name
        self.upsample = F.Upsample(
            name + ".upsample",
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        self.conv = F.Conv2d(name + ".conv", in_channels, out_channels, kernel_size=1)
        self.bn = F.BatchNorm2d(name + ".bn", out_channels)
        self.add = F.Add(name + ".add")
        super().link_op2module()

    def __call__(self, x, x_skip):
        x = self.bn(self.conv(self.upsample(x)))
        return self.add(x, x_skip)


class Bottleneck(SimNN.Module):
    """TTSim SimNN bottleneck residual block."""

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        self.name = "bottleneck"
        out_channels = out_channels or in_channels
        bottleneck_channels = in_channels // 2
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        self.conv_down = F.Conv2d(
            self.name + ".conv_down", in_channels, bottleneck_channels, kernel_size=1
        )
        self.bn_down = F.BatchNorm2d(self.name + ".bn_down", bottleneck_channels)
        self.relu_down = F.Relu(self.name + ".relu_down")

        if upsample:
            self.conv_mid = F.ConvTranspose2d(
                self.name + ".conv_mid",
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                stride=2,
            )
        elif downsample:
            self.conv_mid = F.Conv2d(
                self.name + ".conv_mid",
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding_size,
            )
        else:
            self.conv_mid = F.Conv2d(
                self.name + ".conv_mid",
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=kernel_size,
                padding=padding_size,
            )
        self.bn_mid = F.BatchNorm2d(self.name + ".bn_mid", bottleneck_channels)
        self.relu_mid = F.Relu(self.name + ".relu_mid")

        self.conv_up = F.Conv2d(
            self.name + ".conv_up", bottleneck_channels, out_channels, kernel_size=1
        )
        self.bn_up = F.BatchNorm2d(self.name + ".bn_up", out_channels)
        self.relu_up = F.Relu(self.name + ".relu_up")

        self.add = F.Add(self.name + ".add")

        # Projection
        self.has_proj = not (
            out_channels == in_channels and not downsample and not upsample
        )
        if self.has_proj:
            self.proj_conv = F.Conv2d(
                self.name + ".proj_conv", in_channels, out_channels, kernel_size=1
            )
            self.proj_bn = F.BatchNorm2d(self.name + ".proj_bn", out_channels)

        super().link_op2module()

    def __call__(self, x):
        residual = x
        x = self.relu_down(self.bn_down(self.conv_down(x)))
        x = self.relu_mid(self.bn_mid(self.conv_mid(x)))
        x = self.relu_up(self.bn_up(self.conv_up(x)))
        if self.has_proj:
            residual = self.proj_bn(self.proj_conv(residual))
        return self.add(x, residual)
