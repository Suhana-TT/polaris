# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import math
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_common import Conv
from workloads.segformer.tt.segformer_mlp import TtSegformerMLP


def upsample_nearest_2x(tensor, device):
    """
    Upsample by 2x using nearest neighbor interpolation.
    
    Input: [batch, height, width, channels]
    Output: [batch, height*2, width*2, channels]
    
    This is done by repeating each element along height and width dimensions.
    """
    shape = tensor.shape
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    
    # Method: Use reshape and concat operations
    # Step 1: Reshape to insert dimension for height upsampling
    # [batch, height, width, channels] -> [batch, height, 1, width, channels]
    tensor = ttnn.reshape(tensor, (batch, height, 1, width, channels))
    
    # Step 2: Repeat along the new dimension (height direction)
    # Use concat with axis parameter (not dim)
    tensor = ttnn.concat(tensor, tensor, axis=2)  # [batch, height, 2, width, channels]
    
    # Step 3: Reshape to merge height dimension
    tensor = ttnn.reshape(tensor, (batch, height * 2, width, channels))
    
    # Step 4: Insert dimension for width upsampling
    tensor = ttnn.reshape(tensor, (batch, height * 2, width, 1, channels))
    
    # Step 5: Repeat along width direction
    tensor = ttnn.concat(tensor, tensor, axis=3)  # [batch, height*2, width, 2, channels]
    
    # Step 6: Reshape to final shape
    tensor = ttnn.reshape(tensor, (batch, height * 2, width * 2, channels))
    
    return tensor


def upsample_nearest(tensor, device, scale_factor):
    """
    Upsample using nearest neighbor interpolation.
    
    Args:
        tensor: Input tensor [batch, height, width, channels]
        device: Device handle
        scale_factor: Tuple (scale_h, scale_w) - must be powers of 2
    
    Returns:
        Upsampled tensor [batch, height*scale_h, width*scale_w, channels]
    """
    scale_h, scale_w = scale_factor
    
    # Apply 2x upsampling repeatedly
    current = tensor
    
    # Calculate iterations needed
    h_iterations = int(math.log2(scale_h)) if scale_h > 1 else 0
    w_iterations = int(math.log2(scale_w)) if scale_w > 1 else 0
    
    # For simplicity, we'll do combined 2x upsampling
    # This works when scale_h == scale_w and both are powers of 2
    iterations = max(h_iterations, w_iterations)
    
    for _ in range(iterations):
        current = upsample_nearest_2x(current, device)
    
    return current


class TtSegformerDecodeHead:
    def __init__(self, config, parameters):
        self.config = config
        
        # Linear layers which will unify the channel dimension of each of the 
        # encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = TtSegformerMLP(
                parameters=parameters["linear_c"][i],
                input_dim=config.hidden_sizes[i],
                output_dim=config.decoder_hidden_size,
            )
            mlps.append(mlp)
        self.linear_c = mlps
        
        # Fuse conv: combines all decoder features
        # NOTE: ReLU activation will be applied separately in __call__
        self.linear_fuse = Conv(
            conv_params=[1, 1, 0, 0],
            parameters=parameters["linear_fuse"],
        )
        
        # Classifier: final segmentation head
        self.classifier = Conv(
            conv_params=[1, 1, 0, 0],
            parameters=parameters["classifier"],
        )
    
    def __call__(self, device, encoder_hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Decode head forward pass.
        
        Args:
            device: Device handle
            encoder_hidden_states: Tuple of hidden states from each encoder block
                Each tensor has shape [batch, seq_len, channels]
        
        Returns:
            logits: Segmentation logits [batch, num_classes, height, width] (NCHW)
        """
        batch_size = encoder_hidden_states[-1].shape[0]
        target_size = 128
        
        all_hidden_states = []
        
        # Process each encoder hidden state
        for idx, (encoder_hidden_state, mlp) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            seq_len = encoder_hidden_state.shape[-2]
            height = width = int(math.sqrt(seq_len))
            
            # Apply MLP to unify channel dimensions
            encoder_hidden_state = mlp(device, encoder_hidden_state)
            
            # Reshape to spatial format [batch, height, width, channels]
            encoder_hidden_state = ttnn.reshape(
                encoder_hidden_state, 
                (batch_size, height, width, -1)
            )
            
            # Upsample to target size using nearest neighbor interpolation
            if height != target_size or width != target_size:
                scale_h = target_size // height
                scale_w = target_size // width
                
                # Use manual nearest neighbor upsampling
                encoder_hidden_state = upsample_nearest(
                    encoder_hidden_state,
                    device,
                    scale_factor=(scale_h, scale_w),
                )
            
            all_hidden_states.append(encoder_hidden_state)
        
        # Concatenate all upsampled features along channel dimension (reversed order)
        # ttnn.concat expects axis parameter (not dim)
        # Reverse order: [3, 2, 1, 0] -> largest to smallest encoder block
        concated_tensor = ttnn.concat(
            all_hidden_states[3],  # Block 3: 256 channels (upsampled from 16x16)
            all_hidden_states[2],  # Block 2: 256 channels (upsampled from 32x32)
            all_hidden_states[1],  # Block 1: 256 channels (upsampled from 64x64)
            all_hidden_states[0],  # Block 0: 256 channels (128x128, no upsample)
            axis=3
        )
        
        # Conv expects NCHW format, but we have NHWC
        # Permute: [batch, height, width, channels] -> [batch, channels, height, width]
        concated_tensor = ttnn.permute(concated_tensor, (0, 3, 1, 2))
        
        # Apply fuse convolution
        hidden_states, _, _ = self.linear_fuse(device, concated_tensor)
        
        # Apply ReLU activation (done separately instead of in Conv)
        hidden_states = ttnn.relu(hidden_states)
        
        # Apply classifier
        logits, _, _ = self.classifier(device, hidden_states)
        
        return logits