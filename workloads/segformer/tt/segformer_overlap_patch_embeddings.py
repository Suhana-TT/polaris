# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_common import Conv

class TtSegformerOverlapPatchEmbeddings:
    """Construct the overlapping patch embeddings."""
    
    def __init__(self, parameters, stride, patch_size):
        """
        Initialize OverlapPatchEmbeddings.
        
        Args:
            parameters: Dictionary containing:
                - proj: parameters for Conv layer
                - layer_norm: parameters for LayerNorm (weight, bias)
            stride: Stride for convolution
            patch_size: Size of patches
        """
        self.stride = stride
        self.patch_size = patch_size
        self.parameters = parameters
        
        # Initialize Conv projection layer
        # Conv args: [stride_h, stride_w, padding_h, padding_w]
        self.proj = Conv(
            conv_params=[stride, stride, patch_size // 2, patch_size // 2],
            parameters=parameters["proj"]
        )
    
    def __call__(
        self,
        device,
        pixel_values: ttnn.Tensor,
    ):
        """
        Forward pass.
        
        Args:
            device: Device handle
            pixel_values: Input tensor [batch, channels, height, width]
            
        Returns:
            tuple: (embeddings, input_height, input_width)
                - embeddings: Output tensor after projection and layer norm
                - input_height: Height of feature map after projection
                - input_width: Width of feature map after projection
        """
        # Apply convolution projection
        # Output will be in NCHW format: [batch, out_channels, H, W]
        embeddings, input_height, input_width = self.proj(device, pixel_values)
        
        # Reshape from NCHW to NHWC for layer_norm
        # [N, C, H, W] -> [N, H, W, C]
        embeddings = ttnn.permute(embeddings, (0, 2, 3, 1))
        
        # Now embeddings is [N, H, W, C] and layer_norm will normalize over C (last dim)
        embeddings = ttnn.layer_norm(
            embeddings,
            weight=self.parameters["layer_norm"]["weight"],
            bias=self.parameters["layer_norm"]["bias"],
        )
        
        return embeddings, input_height, input_width