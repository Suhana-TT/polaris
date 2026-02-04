#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np

class BasicBlock(SimNN.Module):
    expansion = 1

    def __init__(self, objname, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.name = objname

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1",
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1  = F.BatchNorm2d(f"{self.name}_bn1", planes)
        self.relu1 = F.Relu(f"{self.name}_relu1")

        self.conv2 = F.Conv2d(
            f"{self.name}_conv2",
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = F.BatchNorm2d(f"{self.name}_bn2", planes)
        self.relu2 = F.Relu(f"{self.name}_relu2")
        self.downsample = downsample
        self.stride     = stride

        super().link_op2module()

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)  
        out = out + residual
        out = self.relu2(out)
        return out
     
class Bottleneck(SimNN.Module):
    expansion = 4

    def __init__(self, objname, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.name = objname

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1",
            inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.bn1  = F.BatchNorm2d(f"{self.name}_bn1", planes)
        self.relu1 = F.Relu(f"{self.name}_relu1")
        self.conv2 = F.Conv2d(
            f"{self.name}_conv2", planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn2  = F.BatchNorm2d(f"{self.name}_bn2", planes)
        self.relu2 = F.Relu(f"{self.name}_relu2")
        self.conv3 = F.Conv2d(
            f"{self.name}_conv3", planes, planes * Bottleneck.expansion, kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.bn3  = F.BatchNorm2d(f"{self.name}_bn3", planes * Bottleneck.expansion)
        self.relu_res = F.Relu(f"{self.name}_relu_res")
        self.downsample = downsample
        self.stride     = stride
        super().link_op2module()

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return self.relu_res(out)
    
class Downsample(SimNN.Module):
    def __init__(self, objname, inplanes, outplanes, stride):
        super().__init__()
        self.name = objname

        self.conv = F.Conv2d(
            f"{self.name}_conv",
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn = F.BatchNorm2d(f"{self.name}_bn", outplanes)

        super().link_op2module()

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BBoxTransform(SimNN.Module):
    def __init__(self, objname):
        super().__init__()
        self.name = objname
        self.std_values = [0.1, 0.1, 0.2, 0.2]
        self.mean_values = [0.0, 0.0, 0.0, 0.0]

        # Pre-create all TTSIM ops used inside this module so they are registered once.
        self.d_split = F.Split(f"{self.name}_d_split", axis=2, count=4)
        self.b_split = F.Split(f"{self.name}_b_split", axis=2, count=4)

        self.exp_w = F.Exp(f"{self.name}_exp_w")
        self.exp_h = F.Exp(f"{self.name}_exp_h")

        self.concat = F.ConcatX(f"{self.name}_concat", axis=2)

        super().link_op2module()

    def __call__(self, boxes, deltas):
        boxes.link_module = self
        deltas.link_module = self

        half = F._from_data(f"{self.name}_half", np.array([0.5], dtype=np.float32))

        s0 = F._from_data(f"{self.name}_s0", np.array([self.std_values[0]], dtype=np.float32))
        s1 = F._from_data(f"{self.name}_s1", np.array([self.std_values[1]], dtype=np.float32))
        s2 = F._from_data(f"{self.name}_s2", np.array([self.std_values[2]], dtype=np.float32))
        s3 = F._from_data(f"{self.name}_s3", np.array([self.std_values[3]], dtype=np.float32))

        m0 = F._from_data(f"{self.name}_m0", np.array([self.mean_values[0]], dtype=np.float32))
        m1 = F._from_data(f"{self.name}_m1", np.array([self.mean_values[1]], dtype=np.float32))
        m2 = F._from_data(f"{self.name}_m2", np.array([self.mean_values[2]], dtype=np.float32))
        m3 = F._from_data(f"{self.name}_m3", np.array([self.mean_values[3]], dtype=np.float32))

        d_list = self.d_split(deltas)
        dx, dy, dw, dh = d_list[0], d_list[1], d_list[2], d_list[3]
        for d in [dx, dy, dw, dh]:
            d.link_module = self

        b_list = self.b_split(boxes)
        x1_a, y1_a, x2_a, y2_a = b_list[0], b_list[1], b_list[2], b_list[3]
        for b in [x1_a, y1_a, x2_a, y2_a]:
            b.link_module = self

        widths = x2_a - x1_a
        heights = y2_a - y1_a
        ctr_x = x1_a + (widths * half)
        ctr_y = y1_a + (heights * half)

        dx_n = dx * s0 + m0
        dy_n = dy * s1 + m1
        dw_n = dw * s2 + m2
        dh_n = dh * s3 + m3

        p_ctr_x = ctr_x + (dx_n * widths)
        p_ctr_y = ctr_y + (dy_n * heights)
        p_w = self.exp_w(dw_n) * widths
        p_h = self.exp_h(dh_n) * heights

        px1 = p_ctr_x - (p_w * half)
        py1 = p_ctr_y - (p_h * half)
        px2 = p_ctr_x + (p_w * half)
        py2 = p_ctr_y + (p_h * half)

        out = self.concat(px1, py1, px2, py2)
        out.link_module = self
        return out
    
class ClipBoxes(SimNN.Module):
    def __init__(self, objname, max_x=None, max_y=None):
        super().__init__()
        self.name = objname

        self.max_x = float(max_x) if max_x is not None else 607.0
        self.max_y = float(max_y) if max_y is not None else 607.0

        self.b_split_clip = F.Split(f"{self.name}_b_split_clip", axis=2, count=4)

        self.clip_x1 = F.Clip(f"{self.name}_clip_x1", min=0.0, max=self.max_x)
        self.clip_y1 = F.Clip(f"{self.name}_clip_y1", min=0.0, max=self.max_y)
        self.clip_x2 = F.Clip(f"{self.name}_clip_x2", min=0.0, max=self.max_x)
        self.clip_y2 = F.Clip(f"{self.name}_clip_y2", min=0.0, max=self.max_y)

        self.concat_clip = F.ConcatX(f"{self.name}_concat_clip", axis=2)

        super().link_op2module()

    def __call__(self, boxes, img):
        boxes.link_module = self
        # NOTE: `img` is accepted for API compatibility with PyTorch reference implementations
        # (the original signature is `__call__(boxes, img)`), but is not used here because
        # clipping bounds are configured via `max_x` and `max_y` in `__init__`.
        b_list = self.b_split_clip(boxes)
        bx1, by1, bx2, by2 = b_list[0], b_list[1], b_list[2], b_list[3]
        for b in [bx1, by1, bx2, by2]:
            b.link_module = self

        px1 = self.clip_x1(bx1)
        py1 = self.clip_y1(by1)
        px2 = self.clip_x2(bx2)
        py2 = self.clip_y2(by2)

        out = self.concat_clip(px1, py1, px2, py2)
        out.link_module = self
        return out