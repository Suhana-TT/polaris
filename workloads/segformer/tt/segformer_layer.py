# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_attention import TtSegformerAttention
from workloads.segformer.tt.segformer_mix_ffn import TtSegformerMixFFN

class TtSegformerLayer:
    def __init__(self, name, hidden_size, num_attention_heads, sequence_reduction_ratio, parameters, mlp_ratio):
        self.name = name
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.sequence_reduction_ratio = sequence_reduction_ratio
        self.mlp_ratio = mlp_ratio
        self.parameters = parameters
        
        # Attention submodule
        self.attention = TtSegformerAttention(
            name=f"{self.name}_attention",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters["attention"],
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        
        # MLP submodule
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TtSegformerMixFFN(
            name=f"{self.name}_mlp",
            parameters=parameters["mlp"],
            config=None,
            in_features=hidden_size,
            hidden_features=mlp_hidden_size,
            out_features=hidden_size,
        )
        
    def __call__(self, device, hidden_states, height: int, width: int, output_attentions=False):
        """
        Forward pass: 
        1. LayerNorm -> Attention -> Residual
        2. LayerNorm -> MLP -> Residual
        
        Args:
            device: Device handle
            hidden_states: Input tensor [batch, seq_len, hidden_size] or [batch, height, width, channels]
            height: Spatial height
            width: Spatial width
            output_attentions: Whether to return attention weights
            
        Returns:
            tuple: (layer_output, [attention_weights if output_attentions])
        """
        # Get input shape info
        input_shape = hidden_states.shape
        batch_size = input_shape[0]
        
        # Determine if input is 4D (from patch embedding) or 3D (sequence format)
        is_4d_input = len(input_shape) == 4
        
        if is_4d_input:
            # Input is [batch, height, width, channels]
            num_channels = input_shape[3]
            seq_len = height * width
            # Flatten to [batch, seq_len, channels] for processing
            hidden_states_3d = ttnn.reshape(hidden_states, (batch_size, seq_len, num_channels))
        else:
            # Input is already [batch, seq_len, hidden_size]
            hidden_states_3d = hidden_states
            seq_len = input_shape[1]
        
        # --- Attention Block ---
        # Layer norm 1
        normalized_hidden_states = ttnn.layer_norm(
            hidden_states_3d,
            weight=self.parameters["layer_norm_1"]["weight"],
            bias=self.parameters["layer_norm_1"]["bias"],
        )
        
        # Self-attention
        self_attention_outputs = self.attention(
            normalized_hidden_states,    # hidden_states (positional)
            height,                      # height (positional)
            width,                       # width (positional)
            output_attentions=output_attentions,  # keyword
            device=device, 
        )
        
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # attention weights if output_attentions=True
        
        # Ensure attention_output has correct shape for residual
        attention_output_shape = attention_output.shape
        if len(attention_output_shape) == 4:
            # Attention output is [1, 1, seq_len, hidden] - reshape to [batch, seq_len, hidden]
            attention_output = ttnn.reshape(attention_output, (batch_size, seq_len, self.hidden_size))
        
        # Residual connection (both should be [batch, seq_len, hidden_size])
        hidden_states_3d = ttnn.add(attention_output, hidden_states_3d)
        
        # --- MLP Block ---
        # Layer norm 2
        normalized_hidden_states = ttnn.layer_norm(
            hidden_states_3d,
            weight=self.parameters["layer_norm_2"]["weight"],
            bias=self.parameters["layer_norm_2"]["bias"],
        )
        
        # MLP (Feed-forward)
        mlp_output = self.mlp(
            device,
            normalized_hidden_states,
            height,
            width,
        )
        
        # Ensure mlp_output has correct shape for residual
        mlp_output_shape = mlp_output.shape
        if len(mlp_output_shape) == 4:
            # MLP output might be [1, 1, seq_len, hidden] - reshape to [batch, seq_len, hidden]
            mlp_output = ttnn.reshape(mlp_output, (batch_size, seq_len, self.hidden_size))
        
        # Residual connection
        layer_output = ttnn.add(mlp_output, hidden_states_3d)
        
        # Return output (keep as 3D [batch, seq_len, hidden_size])
        outputs = (layer_output,) + outputs
        return outputs