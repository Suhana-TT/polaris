# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ttsim_utils.py — Pure Python 3.13 / numpy equivalents of all mmcv/mmdet helpers.
No torch, no mmcv imports. Used across all UniAD ttsim converted files.
"""

import numpy as np
import math
import warnings
from functools import wraps
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ---------------------------------------------------------------------------
# Identity decorators (mmcv runtime decorators — not needed in sim)
# ---------------------------------------------------------------------------
def force_fp32(apply_to=None):
    if callable(apply_to):
        return apply_to  # used as bare @force_fp32
    return lambda fn: fn


def auto_fp16(apply_to=None):
    if callable(apply_to):
        return apply_to
    return lambda fn: fn


def deprecated_api_warning(mapping, cls_name=""):
    return lambda fn: fn


# ---------------------------------------------------------------------------
# Shape utils
# ---------------------------------------------------------------------------
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


class ConfigDict(dict):
    """Plain dict replacement for mmcv.ConfigDict."""

    pass


# ---------------------------------------------------------------------------
# Weight init — no-ops for SimNN (params are shape-only)
# ---------------------------------------------------------------------------
def xavier_init(module, gain=1, bias=0, distribution="uniform"):
    pass


def constant_init(module, val, bias=0):
    pass


def normal_init(module, mean=0, std=1, bias=0):
    pass


def kaiming_init(module, mode="fan_in", nonlinearity="relu"):
    pass


# ---------------------------------------------------------------------------
# Bbox transforms (pure numpy)
# ---------------------------------------------------------------------------
def bbox_cxcywh_to_xyxy(bboxes):
    """cx,cy,w,h → x1,y1,x2,y2"""
    bboxes = np.asarray(bboxes)
    x1 = bboxes[..., 0] - bboxes[..., 2] / 2
    y1 = bboxes[..., 1] - bboxes[..., 3] / 2
    x2 = bboxes[..., 0] + bboxes[..., 2] / 2
    y2 = bboxes[..., 1] + bboxes[..., 3] / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def bbox_xyxy_to_cxcywh(bboxes):
    """x1,y1,x2,y2 → cx,cy,w,h"""
    bboxes = np.asarray(bboxes)
    cx = (bboxes[..., 0] + bboxes[..., 2]) / 2
    cy = (bboxes[..., 1] + bboxes[..., 3]) / 2
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    return np.stack([cx, cy, w, h], axis=-1)


def bbox3d2result(bboxes, scores, labels):
    return dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)


# ---------------------------------------------------------------------------
# Functional
# ---------------------------------------------------------------------------
def multi_apply(func, *args, **kwargs):
    return tuple(map(list, zip(*[func(*_args, **kwargs) for _args in zip(*args)])))


def reduce_mean(tensor):
    """No-op: no distributed in sim."""
    return tensor


def inverse_sigmoid(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.clip(x, eps, 1 - eps) / np.clip(1 - x, eps, 1 - eps))


# ---------------------------------------------------------------------------
# No-op registry (replaces mmcv/mmdet registries)
# ---------------------------------------------------------------------------
class _NoOpRegistry:
    def register_module(self, name=None, force=False, module=None):
        if module is not None:
            return module
        return lambda cls: cls


ATTENTION = _NoOpRegistry()
TRANSFORMER_LAYER = _NoOpRegistry()
TRANSFORMER_LAYER_SEQUENCE = _NoOpRegistry()
HEADS = _NoOpRegistry()
TRANSFORMER = _NoOpRegistry()
BBOX_ASSIGNERS = _NoOpRegistry()
BBOX_SAMPLERS = _NoOpRegistry()
DETECTORS = _NoOpRegistry()


# ---------------------------------------------------------------------------
# Builder stubs — training-only
# ---------------------------------------------------------------------------
def build_loss(cfg):
    raise NotImplementedError("build_loss: training-only")


def build_assigner(cfg):
    raise NotImplementedError("build_assigner: training-only")


def build_sampler(cfg, context=None):
    raise NotImplementedError("build_sampler: training-only")


def build_match_cost(cfg):
    raise NotImplementedError("build_match_cost: training-only")


def build_transformer_layer(cfg):
    raise NotImplementedError("build_transformer_layer: use SimNN module directly")


def build_transformer_layer_sequence(cfg):
    raise NotImplementedError(
        "build_transformer_layer_sequence: use SimNN module directly"
    )


# ---------------------------------------------------------------------------
# Norm builder — returns SimNN layer
# ---------------------------------------------------------------------------
def build_norm_layer(cfg, num_features, postfix=""):
    layer_type = cfg.get("type", "LN") if isinstance(cfg, dict) else "LN"
    name = f"norm{postfix}"
    if layer_type in ("LN", "LayerNorm"):
        return name, F.LayerNorm(name, num_features)
    elif layer_type in ("BN", "BN2d", "BatchNorm2d"):
        return name, F.BatchNorm2d(name, num_features)
    else:
        return name, F.LayerNorm(name, num_features)
