# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_common import TtsimConv
from workloads.segformer.tt.segformer_mlp import TtsimSegformerMLP

class TtsimSegformerDecodeHead:
    def __init__(self, name: str, config, parameters):
        self.name = name
        self.config = config

        # One MLP per encoder stage
        self.linear_c = []
        for i in range(config.num_encoder_blocks):
            self.linear_c.append(
                TtsimSegformerMLP(
                    name=f"{self.name}_linear_c_{i}",
                    parameters=parameters["linear_c"][i],
                )
            )

        # 1x1 conv fuse
        self.linear_fuse = TtsimConv(
            name=f"{self.name}_linear_fuse",
            strides=[1, 1, 0, 0],
            parameters=parameters["linear_fuse"],
            kernel_size=[1, 1],
            groups=1,
        )

        self.relu = F.Relu(f"{self.name}_relu")

        # 1x1 classifier
        self.classifier = TtsimConv(
            name=f"{self.name}_classifier",
            strides=[1, 1, 0, 0],
            parameters=parameters["classifier"],
            kernel_size=[1, 1],
            groups=1,
        )

        # Pairwise concat ops on channel dim for NCHW
        self.concat1 = F.ConcatX(f"{self.name}_concat1", axis=1)
        self.concat2 = F.ConcatX(f"{self.name}_concat2", axis=1)
        self.concat3 = F.ConcatX(f"{self.name}_concat3", axis=1)

    def _get_shape_tensor(self, shape_list, name):
        return T.SimTensor(
            {
                "name": name,
                "data": np.array([int(s) for s in shape_list], dtype=np.int64),
                "shape": [len(shape_list)],
                "dtype": np.int64,
            }
        )

    def __call__(self, encoder_hidden_states):
        target_h, target_w = 128, 128
        all_hidden_states = []

        for i, (state, mlp) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            # state: [B, S, C]
            state = mlp(state)  # [B, S, decoder_hidden_size]

            B, S, C = state.shape
            h = w = int(math.sqrt(S))

            # [B, S, C] -> [B, H, W, C]
            rs_shape = self._get_shape_tensor([B, h, w, C], f"{self.name}_rs_{i}")
            state = F.Reshape(f"{self.name}_reshape_{i}")(state, rs_shape)

            # NHWC -> NCHW for Polaris conv/resize path
            state = F.Transpose(f"{self.name}_to_nchw_{i}", perm=[0, 3, 1, 2])(state)

            # Upsample to 128x128
            if h != target_h or w != target_w:
                scale_h = float(target_h) / h
                scale_w = float(target_w) / w
                state = F.Resize(
                    f"{self.name}_resize_{i}",
                    scale_factor=[scale_h, scale_w],
                    mode="bilinear",
                )(state)

            all_hidden_states.append(state)

        # Reverse order like TT-Metal, but pairwise for Polaris stability
        fused_tensor = self.concat1(all_hidden_states[3], all_hidden_states[2])
        fused_tensor = self.concat2(fused_tensor, all_hidden_states[1])
        fused_tensor = self.concat3(fused_tensor, all_hidden_states[0])

        hidden_states = self.linear_fuse(fused_tensor)
        hidden_states = self.relu(hidden_states)
        logits = self.classifier(hidden_states)

        return logits