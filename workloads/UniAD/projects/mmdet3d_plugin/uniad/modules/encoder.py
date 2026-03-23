# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
TTSim: modules/encoder.py
BEVFormerEncoder is fully implemented in transformer.py — re-exported here.
BEVFormerLayer is covered by BEVFormerEncoderLayer in transformer.py.
get_reference_points and point_sampling are pure numpy (used in data prep).
"""

import numpy as np
from .transformer import (
    BEVFormerEncoder,
    BEVFormerEncoderLayer as BEVFormerLayer,
)  # noqa: F401


def get_reference_points(
    H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, dtype=np.float32
):
    """
    Pure numpy: Get the reference points used in SCA and TSA.
    Args:
        H, W: spatial shape of bev.
        Z: height of pillar.
        num_points_in_pillar: number of points in each pillar.
        dim: '3d' or '2d'.
        bs: batch size.
    """
    if dim == "3d":
        zs = np.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype) / Z
        xs = np.linspace(0.5, W - 0.5, W, dtype=dtype) / W
        ys = np.linspace(0.5, H - 0.5, H, dtype=dtype) / H
        zs, xs, ys = np.meshgrid(zs, xs, ys, indexing="ij")  # (nP, W, H)
        ref_3d = np.stack([xs, ys, zs], axis=-1)  # (nP, W, H, 3)
        ref_3d = ref_3d.transpose(0, 3, 1, 2)  # (nP, 3, W, H)
        ref_3d = ref_3d.reshape(num_points_in_pillar, 3, -1).transpose(
            2, 0, 1
        )  # (H*W, nP, 3)
        ref_3d = np.tile(ref_3d[None], (bs, 1, 1, 1))  # (bs, H*W, nP, 3)
        return ref_3d
    elif dim == "2d":
        ref_y, ref_x = np.meshgrid(
            np.linspace(0.5, H - 0.5, H, dtype=dtype) / H,
            np.linspace(0.5, W - 0.5, W, dtype=dtype) / W,
            indexing="ij",
        )
        ref_2d = np.stack([ref_x, ref_y], axis=-1)  # (H, W, 2)
        ref_2d = ref_2d.reshape(-1, 1, 2)  # (H*W, 1, 2)
        ref_2d = np.tile(ref_2d[None], (bs, 1, 1, 1))  # (bs, H*W, 1, 2)
        return ref_2d


def point_sampling(reference_points, pc_range, img_metas):
    """
    Pure numpy: project reference_points (3D) to camera coordinates.
    Used during BEV feature extraction.
    """
    # This is a data-prep utility — kept as numpy passthrough
    # Full implementation depends on camera calibration from img_metas
    raise NotImplementedError("point_sampling: use uniad_track.py ttsim implementation")
