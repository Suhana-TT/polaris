# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
TTSim stub: SpatialCrossAttention and MSDeformableAttention3D are implemented
in transformer.py as _DeformCrossAttn within BEVFormerEncoder.
This file exists only for import compatibility.
"""

from .transformer import BEVFormerEncoder  # noqa: F401


class SpatialCrossAttention:
    """TTSim stub: implemented in transformer.py as _DeformCrossAttn."""

    pass


class MSDeformableAttention3D:
    """TTSim stub: implemented in transformer.py as _DeformCrossAttn."""

    pass
