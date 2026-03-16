# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import math
import numpy as np

sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_common import TtsimConv
from workloads.segformer.tt.segformer_mlp import TtsimSegformerMLP

class TtsimSegformerDecodeHead:
    def __init__(self, name: str, config, parameters):
        self.name = name
        self.config = config

        # 1. MLPs to unify channel dimensions
        self.linear_c = []
        for i in range(config.num_encoder_blocks):
            self.linear_c.append(
                TtsimSegformerMLP(
                    name=f"{self.name}_linear_c_{i}",
                    parameters=parameters["linear_c"][i]
                )
            )

        # 2. Linear Fuse (1x1 Conv)
        self.linear_fuse = TtsimConv(
            name=f"{self.name}_linear_fuse",
            strides=[1, 1, 0, 0], 
            parameters=parameters["linear_fuse"],
            kernel_size=[1, 1],
            groups=1
        )

        # 3. Classifier (1x1 Conv)
        self.classifier = TtsimConv(
            name=f"{self.name}_classifier",
            strides=[1, 1, 0, 0],
            parameters=parameters["classifier"],
            kernel_size=[1, 1],
            groups=1
        )

    def _get_shape_tensor(self, shape_list, name):
        return T.SimTensor({
            "name": name,
            "data": np.array([int(s) for s in shape_list], dtype=np.int64),
            "shape": [len(shape_list)],
            "dtype": np.int64
        })

    def __call__(self, encoder_hidden_states):
        batch_size = int(encoder_hidden_states[0].shape[0])
        target_h, target_w = 128, 128
        
        all_hidden_states = []
        
        for i, (state, mlp) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            # A. Unify Channel Dimension [B, S, C] -> [B, S, C_decoder]
            state = mlp(state)
            
            # B. Reshape to [B, H, W, C_decoder]
            B, S, C = state.shape
            h = w = int(math.sqrt(S))
            rs_shape = self._get_shape_tensor([B, h, w, C], f"{self.name}_rs_{i}")
            state = F.Reshape(f"{self.name}_reshape_{i}")(state, rs_shape)
            
            # C. Transpose to [B, C_decoder, H, W] for Upsample and Conv
            state = F.Transpose(f"{self.name}_tr_{i}", perm=[0, 3, 1, 2])(state)

            # D. Upsample to target resolution using scale_factor
            if h != target_h or w != target_w:
                scale_h = float(target_h) / h
                scale_w = float(target_w) / w
                state = F.Resize(
                    f"{self.name}_resize_{i}", 
                    scale_factor=[scale_h, scale_w], 
                    mode="bilinear"
                )(state)
            
            all_hidden_states.append(state)

        # E. Concatenate along the Channel dimension (dim=1 in NCHW is Z)
        # FIX: Changed F.Concat to F.ConcatZ
        fused_tensor = F.ConcatX(f"{self.name}_concat", axis=1)(*all_hidden_states[::-1])

        # F. Fuse and Classify
        hidden_states = self.linear_fuse(fused_tensor)
        hidden_states = F.Relu(f"{self.name}_relu")(hidden_states)
        logits = self.classifier(hidden_states)

        return logits