# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_overlap_patch_embeddings import TtSegformerOverlapPatchEmbeddings

def create_polaris_tensor(numpy_array, device):
    """Convert numpy array to Polaris ttnn.Tensor and move to device."""
    tensor = ttnn.as_tensor(numpy_array.astype(np.float32))
    tensor = ttnn.to_device(tensor, device)
    return tensor

# --- Mock Classes ---
class MockConv2d:
    """Simulates torch.nn.Conv2d for patch embedding"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
        self.bias = np.random.randn(out_channels).astype(np.float32)

class MockLayerNorm:
    """Simulates torch.nn.LayerNorm"""
    def __init__(self, hidden_size):
        self.weight = np.random.randn(hidden_size).astype(np.float32)
        self.bias = np.random.randn(hidden_size).astype(np.float32)

class MockOverlapPatchEmbeddings:
    """Simulates SegformerOverlapPatchEmbeddings"""
    def __init__(self, patch_size, stride, num_channels, hidden_size):
        self.proj = MockConv2d(num_channels, hidden_size, patch_size, stride, patch_size // 2)
        self.layer_norm = MockLayerNorm(hidden_size)

def create_custom_mesh_preprocessor(device, padded_in_channels):
    """
    Preprocessor to extract parameters from mock model and move to device.
    Matches TT-Metal structure with proper channel padding.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        # Get original weight and pad if necessary
        weight = model.proj.weight  # [out_channels, in_channels, H, W]
        original_in_channels = weight.shape[1]
        
        # Pad weight tensor if input channels were padded
        if padded_in_channels > original_in_channels:
            pad_size = padded_in_channels - original_in_channels
            # Pad along the in_channels dimension (axis=1)
            weight = np.pad(weight, ((0, 0), (0, pad_size), (0, 0), (0, 0)), mode='constant')
            print(f"       Padded weight in_channels: {original_in_channels} → {padded_in_channels}")
        
        # Conv projection parameters
        parameters["proj"] = {
            "weight": ttnn.to_device(ttnn.as_tensor(weight), device),
            "bias": ttnn.to_device(
                ttnn.as_tensor(model.proj.bias),  # Keep as 1D [C_out]
                device
            ),
        }
        
        # LayerNorm parameters
        parameters["layer_norm"] = {
            "weight": ttnn.to_device(ttnn.as_tensor(model.layer_norm.weight), device),
            "bias": ttnn.to_device(ttnn.as_tensor(model.layer_norm.bias), device),
        }
        
        return parameters
    
    return preprocessor

# --- Test Parameters (Same as TT-Metal) ---
test_cases = [
    # (patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i)
    (7, 4, 3, 32, 1, 512, 512, 0),
    (3, 2, 32, 64, 1, 128, 128, 1),
    (3, 2, 64, 160, 1, 64, 64, 2),
    (3, 2, 160, 256, 1, 32, 32, 3),
]

def test_segformer_overlap_patch_embeddings():
    print("\n" + "=" * 80)
    print("=== Polaris Segformer Overlap Patch Embeddings Tests ===")
    print("=" * 80)
    
    passed_count = 0
    device = ttnn.open_device(device_id=0)
    print(f"Device opened: {device}\n")
    
    for params in test_cases:
        patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i = params
        
        test_name = f"PatchEmb {patch_emb_i} | Channels {num_channels}→{hidden_size} | Size {height}x{width}"
        print(f"\n[TEST] {test_name}")
        print(f"       patch_size={patch_size}, stride={stride}")
        
        try:
            # Calculate padded channels
            CONV2D_MIN_CHANNEL_SIZE = 8
            padded_channels = num_channels
            
            if num_channels < CONV2D_MIN_CHANNEL_SIZE:
                padded_channels = CONV2D_MIN_CHANNEL_SIZE
                print(f"       Padded channels: {num_channels} → {padded_channels}")
            elif num_channels > CONV2D_MIN_CHANNEL_SIZE and num_channels % 32 != 0:
                padded_channels = ((num_channels + 31) // 32) * 32
                print(f"       Padded channels: {num_channels} → {padded_channels}")
            
            # 1. Create mock model with ORIGINAL channels
            mock_model = MockOverlapPatchEmbeddings(
                patch_size=patch_size,
                stride=stride,
                num_channels=num_channels,  # Original channels
                hidden_size=hidden_size,
            )
            
            # 2. Extract parameters using preprocessor (handles weight padding)
            preprocessor = create_custom_mesh_preprocessor(device, padded_channels)
            parameters = preprocessor(mock_model, None, None, None)
            
            # 3. Create input tensor in NCHW format
            # Conv2d expects [batch, channels, height, width]
            input_np = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
            
            # Pad channels if needed
            if padded_channels > num_channels:
                pad_size = padded_channels - num_channels
                # Pad along the channels dimension (axis=1)
                input_np = np.pad(input_np, ((0, 0), (0, pad_size), (0, 0), (0, 0)), mode='constant')
            
            input_tensor = create_polaris_tensor(input_np, device)
            print(f"       Input shape: {input_tensor.shape}")
            
            # 4. Create Polaris model
            model = TtSegformerOverlapPatchEmbeddings(
                parameters=parameters,
                stride=stride,
                patch_size=patch_size,
            )
            
            # 5. Run forward pass
            embeddings, out_height, out_width = model(device, input_tensor)
            
            # 6. Get output shape
            if hasattr(embeddings, 'shape'):
                out_shape = tuple(embeddings.shape)
            else:
                out_shape = tuple(embeddings.data.shape)
            
            # 7. Calculate expected output dimensions
            expected_height = (height + 2 * (patch_size // 2) - patch_size) // stride + 1
            expected_width = (width + 2 * (patch_size // 2) - patch_size) // stride + 1
            expected_seq_len = expected_height * expected_width
            
            # Expected shape can be 3D or 4D
            expected_shape_3d = (batch_size, expected_seq_len, hidden_size)
            expected_shape_4d = (batch_size, 1, expected_seq_len, hidden_size)
            expected_shape_spatial = (batch_size, expected_height, expected_width, hidden_size)
            
            # 8. Verify output
            shape_match = (
                out_shape == expected_shape_3d or 
                out_shape == expected_shape_4d or
                out_shape == expected_shape_spatial
            )
            
            height_match = out_height == expected_height
            width_match = out_width == expected_width
            
            if shape_match and height_match and width_match:
                print(f"[PASS] {test_name}")
                print(f"       Output shape: {out_shape}")
                print(f"       Output H×W: {out_height}×{out_width}")
                passed_count += 1
            else:
                print(f"[FAIL] {test_name}")
                print(f"       Expected shape: {expected_shape_3d} or {expected_shape_4d} or {expected_shape_spatial}")
                print(f"       Got shape: {out_shape}")
                print(f"       Expected H×W: {expected_height}×{expected_width}")
                print(f"       Got H×W: {out_height}×{out_width}")
                
        except Exception as e:
            print(f"[ERROR] {test_name}")
            print(f"        {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Cleanup
    ttnn.close_device(device)
    print("\nDevice closed.")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{len(test_cases)} passed")
    print("=" * 80)
    
    return passed_count == len(test_cases)

if __name__ == "__main__":
    success = test_segformer_overlap_patch_embeddings()
    sys.exit(0 if success else 1)