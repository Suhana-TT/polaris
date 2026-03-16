# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_common import TtsimConv

class TtsimSegformerDWConv:
    def __init__(self, name: str, parameters: dict, dim: int):
        self.name = name
        self.dim = int(dim)
        
        # We explicitly pass the kernel size [3, 3] and groups=dim (Depthwise)
        # to the TtsimConv constructor.
        # .. inside __init__ ..
        # Ensure this line exists in segformer_dwconv.py:
        # segformer_dwconv.py
        self.dwconv = TtsimConv(
            name=f"{self.name}_dwconv",
            strides=[1, 1, 1, 1],
            parameters=parameters["dwconv"],
            kernel_size=[3, 3], # <--- Pass these as simple ints
            groups=int(dim)     # <--- Ensure groups is an int
        )

    def _get_shape_tensor(self, shape_list, name):
        # Force all dimensions to be flat integers
        clean_shape = [int(s) for s in shape_list]
        return T.SimTensor({
            "name": name,
            "data": np.array(clean_shape, dtype=np.int64),
            "shape": [len(clean_shape)],
            "dtype": np.int64
        })

    def __call__(self, x, height: int, width: int):
        B, S, C = x.shape
        # Input volume is B*S*C
        
        # 1. Reshape [B, S, C] to [B, C, H, W]
        # We assume S = H * W
        x = F.Transpose(f"{self.name}_t1", perm=[0, 2, 1])(x)
        x = F.Reshape(f"{self.name}_rs1")(x, self._get_shape_tensor([B, C, height, width], "s1"))
        
        # 2. Conv
        x = self.dwconv(x)
        
        # 3. Flatten back to [B, S, C]
        x = F.Reshape(f"{self.name}_rs2")(x, self._get_shape_tensor([B, C, S], "s2"))
        x = F.Transpose(f"{self.name}_t2", perm=[0, 2, 1])(x)
        return x
