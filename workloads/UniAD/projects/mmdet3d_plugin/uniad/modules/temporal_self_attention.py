# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
TTSim stub: TemporalSelfAttention is implemented in transformer.py as _TSA / BEVFormerEncoder.
This file exists only for import compatibility with code that imports TemporalSelfAttention.
"""

from .transformer import BEVFormerEncoder  # noqa: F401


class TemporalSelfAttention:
    """TTSim stub: TemporalSelfAttention not used in ttsim simulation path.
    Full implementation is in transformer.py as _TSA within BEVFormerEncoder.
    """

    pass
