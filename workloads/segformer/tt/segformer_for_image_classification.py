# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_model import TtSegformerModel


@dataclass
class TtSegFormerImageClassifierOutput:
    """Output class for Segformer Image Classification."""
    loss: Optional[ttnn.Tensor] = None
    logits: Optional[ttnn.Tensor] = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtSegformerForImageClassification:
    """
    Segformer model for image classification.
    
    Takes an image and outputs class logits by:
    1. Passing image through Segformer encoder
    2. Global average pooling on the final encoder output
    3. Linear classifier to produce class logits
    """
    
    def __init__(self, config, parameters):
        """
        Initialize Segformer for Image Classification.
        
        Args:
            config: Model configuration with:
                - num_labels: Number of classification classes
                - hidden_sizes: List of hidden sizes for each encoder block
                - use_return_dict: Whether to return dict or tuple
            parameters: Pre-processed model parameters containing:
                - segformer: Parameters for the Segformer encoder
                - classifier: Parameters for the classification head
                    - weight: [num_labels, hidden_size]
                    - bias: [num_labels]
        """
        self.config = config
        self.num_labels = config.num_labels
        
        # Initialize the Segformer encoder backbone
        self.segformer = TtSegformerModel(config, parameters=parameters["segformer"])
        
        # Store classifier parameters
        self.classifier_weight = parameters["classifier"]["weight"]
        self.classifier_bias = parameters["classifier"]["bias"]
    
    def __call__(
        self,
        device,
        pixel_values: ttnn.Tensor,
        labels: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TtSegFormerImageClassifierOutput]:
        """
        Forward pass for image classification.
        
        Args:
            device: Device handle
            pixel_values: Input image tensor [batch, channels, height, width]
            labels: Optional ground truth labels for loss computation (not implemented)
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dataclass or tuple
        
        Returns:
            TtSegFormerImageClassifierOutput or tuple containing:
                - loss: None (loss computation not implemented)
                - logits: Classification logits [batch, num_labels]
                - hidden_states: Optional tuple of hidden states
                - attentions: Optional tuple of attention weights
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Pass through Segformer encoder
        outputs = self.segformer(
            device,
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get the final encoder output (last hidden state)
        # outputs[0] is the sequence output from the last encoder block
        sequence_output = outputs[0]
        
        # Get batch size from output shape
        batch_size = sequence_output.shape[0]
        
        # Reshape to [batch, seq_len, hidden_size]
        # The last encoder block has hidden_sizes[-1] channels
        sequence_output = ttnn.reshape(
            sequence_output, 
            (batch_size, -1, self.config.hidden_sizes[-1])
        )
        
        # Global average pooling: mean over sequence dimension
        # [batch, seq_len, hidden_size] -> [batch, 1, hidden_size]
        sequence_output = ttnn.mean(sequence_output, dim=1, keepdim=True)
        
        # Squeeze to [batch, hidden_size] or handle single batch case
        # Remove the sequence dimension
        sequence_output = ttnn.squeeze(sequence_output, dim=1)
        
        # Apply classifier: linear projection to num_labels
        # [batch, hidden_size] @ [hidden_size, num_labels] -> [batch, num_labels]
        logits = ttnn.linear(
            sequence_output,
            self.classifier_weight,
            bias=self.classifier_bias,
        )
        
        # Loss computation not implemented (would need labels)
        loss = None
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # Return structured output
        return TtSegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )