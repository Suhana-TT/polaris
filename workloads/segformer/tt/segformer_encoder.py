# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_layer import TtSegformerLayer
from workloads.segformer.tt.segformer_overlap_patch_embeddings import (
    TtSegformerOverlapPatchEmbeddings,
)


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


class TtSegformerEncoder:
    def __init__(
        self,
        config,
        parameters,
    ):
        self.config = config
        
        # Patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                TtSegformerOverlapPatchEmbeddings(
                    parameters=parameters["patch_embeddings"][i],
                    stride=config.strides[i],
                    patch_size=config.patch_sizes[i],
                )
            )
        self.patch_embeddings = embeddings
        
        # Transformer blocks
        blocks = []
        for i in range(config.num_encoder_blocks):
            # Each block consists of layers
            layers = []
            for j in range(config.depths[i]):
                layers.append(
                    TtSegformerLayer(
                        name=f"encoder.block.{i}.layer.{j}",
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        parameters=parameters["block"][i][j],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            blocks.append(layers)
        self.block = blocks
        
        # Store layer norm parameters
        self.layer_norm_params = parameters["layer_norm"]
    
    def __call__(
        self,
        device,
        pixel_values: ttnn.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, TtBaseModelOutput]:
        """
        Forward pass through the encoder.
        
        Args:
            device: Device handle
            pixel_values: Input tensor [batch, channels, height, width] (NCHW)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return a dataclass instead of tuple
            
        Returns:
            TtBaseModelOutput or tuple containing:
                - last_hidden_state: Final hidden state
                - hidden_states: All hidden states (if output_hidden_states=True)
                - attentions: All attention weights (if output_attentions=True)
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        
        batch_size = pixel_values.shape[0]
        hidden_states = pixel_values
        
        # Track current spatial dimensions for reshaping between blocks
        current_height = None
        current_width = None
        current_channels = None
        
        # Process through each encoder block
        for idx, (embedding_layer, block_layer) in enumerate(zip(self.patch_embeddings, self.block)):
            
            # For blocks after the first one, reshape from [B, S, C] to [B, C, H, W]
            if idx > 0:
                # hidden_states is [batch, seq_len, channels] from previous block
                # Need to reshape to [batch, channels, height, width] for patch embedding
                
                # First reshape to [B, H, W, C]
                hidden_states = ttnn.reshape(
                    hidden_states, 
                    (batch_size, current_height, current_width, current_channels)
                )
                # Then permute to [B, C, H, W] for conv2d
                hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
            
            # Step 1: Obtain patch embeddings
            # Input: [B, C, H, W] (NCHW)
            # Output: [B, H', W', C'] (NHWC) after the embedding layer
            hidden_states, height, width = embedding_layer(
                device,
                pixel_values=hidden_states,
            )
            
            # Update current dimensions for next iteration
            current_height = height
            current_width = width
            current_channels = self.config.hidden_sizes[idx]
            
            # Step 2: Send embeddings through transformer layers
            # TtSegformerLayer expects and returns [B, S, C] format
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(
                    device,
                    hidden_states,
                    height,
                    width,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            
            # Step 3: Apply layer norm
            # hidden_states is [B, S, C] after transformer layers
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.layer_norm_params[idx]["weight"],
                bias=self.layer_norm_params[idx]["bias"],
            )
            
            # Store hidden states if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # hidden_states remains as [B, S, C] for the next iteration
            # It will be reshaped at the start of the next block
        
        # Final output handling
        if idx == len(self.patch_embeddings) - 1 and not self.config.reshape_last_stage:
            # Reshape to [batch, height, width, channels] for decoder
            hidden_states = ttnn.reshape(
                hidden_states,
                (batch_size, current_height, current_width, current_channels)
            )
        # else: keep as [batch, seq_len, channels]
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return TtBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )