#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ttsim.front.functional.sim_nn as SimNN

class Anchors(SimNN.Module):
    def __init__(self, objname, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super().__init__()
        self.name = objname

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        else:
            self.strides = strides

        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        else:
            self.sizes = sizes

        if ratios is None:
            self.ratios = np.array([0.5, 1, 2], dtype=np.float32)
        else:
            self.ratios = np.array(ratios, dtype=np.float32)

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], dtype=np.float32)
        else:
            self.scales = np.array(scales, dtype=np.float32)

    def __call__(self, image_h, image_w):
        image_shape_hw = (image_h, image_w)
        anchors = anchors_for_shape(
            image_shape_hw,
            pyramid_levels=self.pyramid_levels,
            ratios=self.ratios,
            scales=self.scales,
            strides=self.strides,
            sizes=self.sizes,
        )
        return anchors.reshape(1, -1, 4)
    

def generate_anchors(base_size=16, ratios=None, scales=None):

    if ratios is None:
        ratios = np.array([0.5, 1.0, 2.0], dtype=np.float32)

    if scales is None:
        scales = np.array([2 ** 0,
                           2 ** (1.0 / 3.0),
                           2 ** (2.0 / 3.0)], dtype=np.float32)

    num_anchors = len(ratios) * len(scales)

    anchors = np.zeros((num_anchors, 4), dtype=np.float32)

    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    areas = anchors[:, 2] * anchors[:, 3]

    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def compute_shape(image_shape_hw, pyramid_levels):
    """
    Compute feature map shapes (H_l,W_l) for each pyramid level.

    image_shape_hw: (H, W) of the input image
    pyramid_levels: list like [3,4,5,6,7]
    """
    image_shape = np.array(image_shape_hw[:2], dtype=np.int64)
    shapes = [(image_shape + 2 ** p - 1) // (2 ** p) for p in pyramid_levels]
    return shapes  # list of (H_l, W_l)

def shift(shape_hw, stride, anchors):
    H, W = shape_hw

    shift_x = (np.arange(0, W) + 0.5) * stride
    shift_y = (np.arange(0, H) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()  # [K,4]

    A = anchors.shape[0]  # number of base anchors
    K = shifts.shape[0]   # number of feature map locations

    # (K,1,4) + (1,A,4) -> (K,A,4) -> (K*A,4)
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((K, 1, 4)))
    all_anchors = all_anchors.reshape((K * A, 4)).astype(np.float32)

    return all_anchors

def anchors_for_shape(
    image_shape_hw,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
):
    """
    Compute all anchors over all pyramid levels for a given image shape (H,W).

    Returns:
        all_anchors: numpy array [A, 4] in (x1,y1,x2,y2) format.
    """
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if strides is None:
        strides = [2 ** p for p in pyramid_levels]

    if sizes is None:
        sizes = [2 ** (p + 2) for p in pyramid_levels]  # same as original

    image_shapes = compute_shape(image_shape_hw, pyramid_levels)

    all_anchors = np.zeros((0, 4), dtype=np.float32)
    for idx, p in enumerate(pyramid_levels):
        base_anchors = generate_anchors(
            base_size=sizes[idx],
            ratios=ratios,
            scales=scales,
        )
        shifted = shift(image_shapes[idx], strides[idx], base_anchors)
        all_anchors = np.concatenate([all_anchors, shifted], axis=0)

    return all_anchors  # [A,4]
