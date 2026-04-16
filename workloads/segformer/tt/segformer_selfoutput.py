# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.ttnn as ttnn


class TtSegformerSelfOutput:
    def __init__(self, name, hidden_size, parameters):
        self.name = name
        self.hidden_size = hidden_size
        
        self.dense_weight = parameters["dense"]["weight"]
        self.dense_bias = parameters["dense"]["bias"]

    def __call__(self, hidden_states):
        # Get input dimensions (for reference, not used for sharding in Polaris)
        if len(hidden_states.shape) == 4:
            batch_size, _, seq_len, hidden_size = hidden_states.shape
        elif len(hidden_states.shape) == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Linear projection (dense layer)
        hidden_states = ttnn.linear(
            hidden_states,
            self.dense_weight,
            bias=self.dense_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        
        hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
        
        return hidden_states