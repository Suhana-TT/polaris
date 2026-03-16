# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_dwconv import TtsimSegformerDWConv

class TtsimSegformerMixFFN:
    def __init__(self, name: str, config, in_features, hidden_features, out_features, parameters):
        self.name = name
        self.dtype = "float32" 

        # 1. Expansion Layer
        self.dense1 = SimNN.Linear(
            name=f"{self.name}_dense1",
            in_features=in_features,
            out_features=hidden_features
        )
        self.dense1.weight = T.SimTensor({
            "name": f"{self.name}_dense1_w", 
            "data": parameters["dense1"]["weight"], 
            "shape": list(parameters["dense1"]["weight"].shape),
            "dtype": self.dtype
        })
        self.dense1.bias = T.SimTensor({
            "name": f"{self.name}_dense1_b", 
            "data": parameters["dense1"]["bias"], 
            "shape": list(parameters["dense1"]["bias"].shape),
            "dtype": self.dtype
        })

        # 2. Depthwise Convolution
        self.dwconv = TtsimSegformerDWConv(
            name=f"{self.name}_dwconv",
            parameters=parameters, 
            dim=hidden_features
        )

        # 3. Activation Function 
        # Pass name as a positional argument to satisfy UniversalOperator
        self.gelu = F.Gelu(f"{self.name}_gelu")

        # 4. Projection Layer
        self.dense2 = SimNN.Linear(
            name=f"{self.name}_dense2",
            in_features=hidden_features,
            out_features=out_features
        )
        self.dense2.weight = T.SimTensor({
            "name": f"{self.name}_dense2_w", 
            "data": parameters["dense2"]["weight"], 
            "shape": list(parameters["dense2"]["weight"].shape),
            "dtype": self.dtype
        })
        self.dense2.bias = T.SimTensor({
            "name": f"{self.name}_dense2_b", 
            "data": parameters["dense2"]["bias"], 
            "shape": list(parameters["dense2"]["bias"].shape),
            "dtype": self.dtype
        })

    def __call__(self, hidden_states, height: int, width: int):
        # Step 1: Expansion
        hidden_states = self.dense1(hidden_states)
        
        # Step 2: Spatial (DWConv)
        hidden_states = self.dwconv(hidden_states, height, width)
        
        # Step 3: Activation
        hidden_states = self.gelu(hidden_states)
        
        # Step 4: Projection
        hidden_states = self.dense2(hidden_states)
        
        return hidden_states
