# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_common import TtsimConv

class TtsimSegformerOverlapPatchEmbeddings:
    def __init__(self, name: str, parameters, stride, patch_size):
        self.name = name
        
        # Extract channel size
        self.hidden_size = int(parameters["layer_norm"]["weight"].shape[0])
        
        # SAFE BROADCASTING: Reshape 1D weights to [1, 1, C] for [B, Seq, C] math
        self.ln_w = T.SimTensor({
            "name": f"{name}_ln_w", 
            "data": parameters["layer_norm"]["weight"].reshape(1, 1, self.hidden_size), 
            "shape": [1, 1, self.hidden_size], 
            "dtype": "float32"
        })
        self.ln_b = T.SimTensor({
            "name": f"{name}_ln_b", 
            "data": parameters["layer_norm"]["bias"].reshape(1, 1, self.hidden_size), 
            "shape": [1, 1, self.hidden_size], 
            "dtype": "float32"
        })
        
        self.proj = TtsimConv(
            name=f"{name}_proj",
            strides=[stride, stride, patch_size // 2, patch_size // 2],
            parameters=parameters["proj"],
            kernel_size=[patch_size, patch_size],
            groups=1
        )

    # Helper to prevent tensor name collisions in the graph
    def _get_shape_tensor(self, shape_list, name):
        return T.SimTensor({
            "name": name,
            "data": np.array([int(s) for s in shape_list], dtype=np.int64),
            "shape": [len(shape_list)],
            "dtype": np.int64
        })

    def __call__(self, pixel_values):
        # 1. Overlapping Convolution
        x = self.proj(pixel_values)
        B = int(x.shape[0])
        C = int(x.shape[1])
        H = int(x.shape[2])
        W = int(x.shape[3])
        
        # 2. Permute [B, C, H, W] -> [B, H, W, C]
        x = F.Transpose(f"{self.name}_pm1", perm=[0, 2, 3, 1])(x)
        
        # 3. Flatten Spatial Dims [B, H, W, C] -> [B, H*W, C]
        rs_shape = self._get_shape_tensor([B, H * W, C], f"{self.name}_rs1")
        x = F.Reshape(f"{self.name}_reshape")(x, rs_shape)
        
        # 4. LayerNorm (Standardize -> Scale -> Shift)
        # We explicitly tell it the normalized shape [C]
        x = F.LayerNorm(f"{self.name}_ln", C)(x)
        x = F.Mul(f"{self.name}_ln_mul")(x, self.ln_w)
        x = F.Add(f"{self.name}_ln_add")(x, self.ln_b)
        
        return x, H, W
