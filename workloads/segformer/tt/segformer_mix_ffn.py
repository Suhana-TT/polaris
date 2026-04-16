# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_dwconv import TtSegformerDWConv

class TtSegformerMixFFN:
    def __init__(self, name: str, parameters, config, in_features, hidden_features, out_features):
        self.name = name
        self.config = config
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.parameters = parameters
        
        # Initialize DWConv submodule
        self.dwconv = TtSegformerDWConv(
            name=f"{self.name}_dwconv",
            parameters=parameters["dwconv"],
            dim=hidden_features
        )
        
    def __call__(self, device, hidden_states, height: int, width: int):
        """
        Forward pass: Dense1 -> DWConv -> Dense2
        """
        # Get input shape
        shape = hidden_states.shape
        
        if len(shape) == 3:
            batch_size, seq_len, num_channels = shape
        elif len(shape) == 4:
            batch_size, _, seq_len, num_channels = shape
            # Reshape 4D to 3D
            hidden_states = ttnn.reshape(hidden_states, (batch_size, seq_len, num_channels))
        else:
            raise ValueError(f"Unexpected hidden_states shape: {shape}")
        
        # First linear layer (dense1): [B, S, in_features] -> [B, S, hidden_features]
        # Manually do matmul instead of ttnn.linear
        # hidden @ weight.T + bias
        weight1_T = ttnn.transpose(self.parameters["dense1"]["weight"], -2, -1)
        hidden_states = ttnn.matmul(hidden_states, weight1_T)
        hidden_states = ttnn.add(hidden_states, self.parameters["dense1"]["bias"])
        
        # Depthwise convolution: [B, S, hidden_features] -> [B, H, W, hidden_features]
        hidden_states, _, _ = self.dwconv(device, hidden_states, height, width)
        
        # Reshape back from [B, H, W, C] to [B, S, C]
        hidden_states = ttnn.reshape(hidden_states, (batch_size, seq_len, self.hidden_features))
        
        # Second linear layer (dense2): [B, S, hidden_features] -> [B, S, out_features]
        weight2_T = ttnn.transpose(self.parameters["dense2"]["weight"], -2, -1)
        hidden_states = ttnn.matmul(hidden_states, weight2_T)
        hidden_states = ttnn.add(hidden_states, self.parameters["dense2"]["bias"])
        
        return hidden_states