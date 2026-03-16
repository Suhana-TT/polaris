# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Add project root to sys.path so 'workloads' can always be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from workloads.segformer.tt.segformer_efficient_selfattention import TtsimSegformerEfficientSelfAttention
from workloads.segformer.tt.segformer_selfoutput import TtsimSegformerSelfOutput

class TtsimSegformerAttention:
    def __init__(self, name: str, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio):
        self.name = name
        
        # Initialize Self Attention (Removed 'config')
        self.self_attention = TtsimSegformerEfficientSelfAttention(
            name=f"{self.name}_self",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters["self"],
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        
        # Initialize Output
        self.output = TtsimSegformerSelfOutput(
            name=f"{self.name}_output",
            hidden_size=hidden_size,
            parameters=parameters["output"]
        )

    def __call__(self, hidden_states, height: int, width: int, output_attentions=False):
        # self_outputs is (attention_output, ...)
        self_outputs = self.self_attention(hidden_states, height, width)
        
        # SelfOutput needs the attention output (self_outputs[0]) AND the original hidden_states
        attention_output = self.output(self_outputs[0], hidden_states)
        
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
