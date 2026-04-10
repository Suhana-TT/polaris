# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import ttsim.front.functional.sim_nn as SimNN
from workloads.segformer.tt.segformer_dwconv import TtsimSegformerDWConv

class TtsimSegformerMixFFN(SimNN.Module):
    def __init__(self, name: str, config, in_features, hidden_features, out_features, parameters):
        super().__init__()
        self.name = name
        self.dtype = "float32"

        # dense1: [B, S, in_features] -> [B, S, hidden_features]
        self.dense1 = SimNN.Linear(
            name=f"{self.name}_dense1",
            in_features=in_features,
            out_features=hidden_features,
            bias=True,
        )

        # TT-Metal preprocess_linear_weight stores [in_features, out_features]
        # SimNN.Linear expects param.data as [out_features, in_features]
        self.dense1.param.data = np.array(parameters["dense1"]["weight"], dtype=np.float32).T.copy()
        self.dense1.bias.data = np.array(parameters["dense1"]["bias"], dtype=np.float32).reshape(-1).copy()

        # DWConv gets only its own sub-dict
        self.dwconv = TtsimSegformerDWConv(
            name=f"{self.name}_dwconv",
            parameters=parameters["dwconv"],
            dim=hidden_features,
        )

        # dense2: [B, S, hidden_features] -> [B, S, out_features]
        self.dense2 = SimNN.Linear(
            name=f"{self.name}_dense2",
            in_features=hidden_features,
            out_features=out_features,
            bias=True,
        )
        self.dense2.param.data = np.array(parameters["dense2"]["weight"], dtype=np.float32).T.copy()
        self.dense2.bias.data = np.array(parameters["dense2"]["bias"], dtype=np.float32).reshape(-1).copy()

        super().link_op2module()

    def __call__(self, hidden_states, height: int, width: int):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.dense2(hidden_states)
        return hidden_states