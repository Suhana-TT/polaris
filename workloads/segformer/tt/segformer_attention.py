# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_efficient_selfattention import TtsimSegformerEfficientSelfAttention
from workloads.segformer.tt.segformer_selfoutput import TtsimSegformerSelfOutput


class TtsimSegformerAttention:
    def __init__(self, name, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio):
        self.name = name
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.sequence_reduction_ratio = sequence_reduction_ratio
        
        self.self_attention = TtsimSegformerEfficientSelfAttention(
            name=f"{name}_self_attention",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters["self"],
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        
        self.output = TtsimSegformerSelfOutput(
            name=f"{name}_output",
            hidden_size=hidden_size,
            parameters=parameters["output"],
        )

    def __call__(
        self,
        hidden_states,
        height: int,
        width: int,
        output_attentions: bool = False,
    ):
        self_outputs = self.self_attention(
            hidden_states,
            height,
            width,
            output_attentions=output_attentions,
        )
        
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]
        ttnn.deallocate(self_outputs[0])
        
        return outputs