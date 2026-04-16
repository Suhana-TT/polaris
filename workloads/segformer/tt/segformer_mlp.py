# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttsim.front.ttnn as ttnn


class TtSegformerMLP:
    """
    Simple MLP for the decode head that projects encoder hidden states
    to a unified decoder hidden size.
    
    This is different from TtSegformerMixFFN which is used in the encoder.
    The MLP performs a simple linear projection: output = input @ weight.T + bias
    """
    
    def __init__(self, parameters, input_dim=None, output_dim=None):
        """
        Args:
            parameters: Dictionary containing 'proj' with 'weight' and 'bias'
            input_dim: Input feature dimension (optional, for documentation)
            output_dim: Output feature dimension (optional, for documentation)
        """
        self.parameters = parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def __call__(self, device, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass: simple linear projection.
        
        Args:
            device: Device handle
            hidden_states: Input tensor [batch, seq_len, input_dim]
        
        Returns:
            Projected tensor [batch, seq_len, output_dim]
        """
        # Get input shape for reference
        input_shape = hidden_states.shape
        
        # Ensure we have 3D input [batch, seq_len, channels]
        if len(input_shape) == 4:
            # [batch, 1, seq_len, channels] -> [batch, seq_len, channels]
            batch_size = input_shape[0]
            seq_len = input_shape[2] if input_shape[1] == 1 else input_shape[1] * input_shape[2]
            channels = input_shape[3] if input_shape[1] == 1 else input_shape[3]
            hidden_states = ttnn.reshape(hidden_states, (batch_size, seq_len, channels))
        
        # Linear projection: hidden_states @ weight.T + bias
        # Weight shape: [output_dim, input_dim] (PyTorch convention)
        # After transpose: [input_dim, output_dim]
        # Result: [batch, seq_len, input_dim] @ [input_dim, output_dim] = [batch, seq_len, output_dim]
        weight_T = ttnn.transpose(self.parameters["proj"]["weight"], -2, -1)
        hidden_states = ttnn.matmul(hidden_states, weight_T)
        hidden_states = ttnn.add(hidden_states, self.parameters["proj"]["bias"])
        
        return hidden_states