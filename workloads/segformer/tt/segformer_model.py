# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_encoder import TtSegformerEncoder


@dataclass
class TtBaseModelOutput:
    last_hidden_state: ttnn.Tensor = None
    hidden_states: tuple = None
    attentions: tuple = None
    
    def __getitem__(self, idx):
        if idx == 0:
            return self.last_hidden_state
        elif idx == 1:
            return self.hidden_states
        elif idx == 2:
            return self.attentions
        else:
            raise IndexError("Index out of range")


class TtSegformerModel:
    def __init__(self, config, parameters):
        self.config = config
        
        # Hierarchical Transformer encoder
        self.encoder = TtSegformerEncoder(config, parameters["encoder"])
    
    def __call__(
        self,
        device,
        pixel_values: ttnn.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TtBaseModelOutput]:
        """
        Forward pass through the Segformer model.
        
        Args:
            device: Device handle
            pixel_values: Input tensor [batch, channels, height, width] (NCHW format)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dataclass instead of tuple
            
        Returns:
            TtBaseModelOutput or tuple containing:
                - last_hidden_state: Final encoder hidden state
                - hidden_states: All hidden states (if output_hidden_states=True)
                - attentions: All attention weights (if output_attentions=True)
        """
        # Handle default values from config
        output_attentions = (
            output_attentions if output_attentions is not None 
            else getattr(self.config, 'output_attentions', False)
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None 
            else getattr(self.config, 'output_hidden_states', False)
        )
        return_dict = (
            return_dict if return_dict is not None 
            else getattr(self.config, 'use_return_dict', True)
        )
        
        # Input is expected in NCHW format [batch, channels, height, width]
        # The encoder handles the channel padding and format conversion internally
        
        # Pass through encoder
        # Note: The encoder expects NCHW format and handles conversion internally
        encoder_outputs = self.encoder(
            device,
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        
        return TtBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )