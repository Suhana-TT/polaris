# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_efficient_selfattention import TtSegformerEfficientSelfAttention
from workloads.segformer.tt.segformer_selfoutput import TtSegformerSelfOutput

class TtSegformerAttention:
    def __init__(self, name, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio):
        """
        Initialize SegformerAttention.
        
        Args:
            name: Layer name for identification
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            parameters: Dictionary containing:
                - self: parameters for EfficientSelfAttention
                - output: parameters for SelfOutput
            sequence_reduction_ratio: Ratio for sequence reduction in attention
        """
        self.name = name
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.sequence_reduction_ratio = sequence_reduction_ratio
        
        # Initialize sub-modules with their respective parameters
        self.self_attention = TtSegformerEfficientSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters["self"],
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        
        self.output = TtSegformerSelfOutput(
            name=f"{name}.output",
            hidden_size=hidden_size,
            parameters=parameters["output"],
        )
    
    def __call__(
        self,
        hidden_states,
        height: int,
        width: int,
        output_attentions: bool = False,
        device=None,  # Add device parameter
    ):
        """
        Forward pass.
        
        Args:
            hidden_states: Input tensor
            height: Height of the feature map
            width: Width of the feature map
            output_attentions: Whether to return attention weights
            device: Device for tensor operations
            
        Returns:
            outputs: Tuple of (attention_output, [attention_weights if output_attentions])
        """
        # Self attention - pass device
        self_outputs = self.self_attention(
            hidden_states=hidden_states,
            height=height,
            width=width,
            output_attentions=output_attentions,
            device=device,  # Pass device to self_attention
        )
        
        # Output projection
        attention_output = self.output(self_outputs[0])
        
        # Combine outputs
        outputs = (attention_output,) + self_outputs[1:]
        
        return outputs 