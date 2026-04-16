# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_decode_head import TtSegformerDecodeHead
from workloads.segformer.tt.segformer_model import TtSegformerModel


@dataclass
class TtSemanticSegmenterOutput:
    """Output class for Segformer Semantic Segmentation."""
    loss: Optional[ttnn.Tensor] = None
    logits: Optional[ttnn.Tensor] = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtSegformerForSemanticSegmentation:
    """
    Segformer model for semantic segmentation.
    
    Architecture:
    1. Segformer encoder backbone - extracts multi-scale features
    2. Decode head - fuses multi-scale features and produces segmentation map
    
    Input: Image [batch, channels, height, width]
    Output: Segmentation logits [batch, num_labels, height/4, width/4]
    """
    
    def __init__(self, config, parameters):
        """
        Initialize Segformer for Semantic Segmentation.
        
        Args:
            config: Model configuration with:
                - num_labels: Number of segmentation classes
                - num_encoder_blocks: Number of encoder stages (typically 4)
                - hidden_sizes: List of hidden sizes for each encoder block
                - decoder_hidden_size: Hidden size in decode head
                - use_return_dict: Whether to return dict or tuple
                - output_hidden_states: Whether to output hidden states
            parameters: Pre-processed model parameters containing:
                - segformer: Parameters for the Segformer encoder
                - decode_head: Parameters for the decode head
        """
        self.config = config
        
        # Initialize the Segformer encoder backbone
        self.segformer = TtSegformerModel(config, parameters=parameters["segformer"])
        
        # Initialize the decode head for segmentation
        self.decode_head = TtSegformerDecodeHead(config, parameters=parameters["decode_head"])
    
    def __call__(
        self,
        device,
        pixel_values: ttnn.Tensor,
        labels: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TtSemanticSegmenterOutput]:
        """
        Forward pass for semantic segmentation.
        
        Args:
            device: Device handle
            pixel_values: Input image tensor [batch, channels, height, width]
            labels: Optional ground truth segmentation labels (not implemented)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dataclass or tuple
        
        Returns:
            TtSemanticSegmenterOutput containing:
                - loss: None (loss computation not implemented)
                - logits: Segmentation logits [batch, num_labels, H/4, W/4]
                - hidden_states: Optional tuple of encoder hidden states
                - attentions: Optional tuple of attention weights
        """
        # Use config defaults if not specified
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None 
            else self.config.output_hidden_states
        )
        
        # Validate labels if provided
        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")
        
        # Pass through Segformer encoder
        # We always need intermediate hidden states for the decode head
        outputs = self.segformer(
            device,
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always need intermediate hidden states
            return_dict=return_dict,
        )
        
        # Get encoder hidden states from all blocks
        # These are the multi-scale features needed by the decode head
        if return_dict:
            encoder_hidden_states = outputs.hidden_states
        else:
            encoder_hidden_states = outputs[1]
        
        # Pass through decode head to get segmentation logits
        # encoder_hidden_states: tuple of 4 tensors from each encoder block
        # logits: [batch, num_labels, height/4, width/4]
        logits = self.decode_head(device, encoder_hidden_states)
        
        # Loss computation not implemented
        loss = None
        
        # Return structured output
        return TtSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )