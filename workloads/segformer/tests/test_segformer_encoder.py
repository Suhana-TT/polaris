# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_encoder import TtSegformerEncoder
from workloads.segformer.tests.test_segformer_layer import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_layer,
    MockSegformerLayer,
)
from workloads.segformer.tests.test_segformer_overlap_patch_embeddings import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_overlap_path,
    MockOverlapPatchEmbeddings,
)


def create_polaris_tensor(numpy_array, device):
    """Convert numpy array to Polaris ttnn.Tensor and move to device."""
    tensor = ttnn.as_tensor(numpy_array.astype(np.float32))
    tensor = ttnn.to_device(tensor, device)
    return tensor


def calculate_padded_channels(num_channels):
    """
    Calculate padded channels for conv2d operations.
    Matches the logic in test_segformer_overlap_patch_embeddings.py
    """
    CONV2D_MIN_CHANNEL_SIZE = 8
    
    if num_channels < CONV2D_MIN_CHANNEL_SIZE:
        return CONV2D_MIN_CHANNEL_SIZE
    elif num_channels > CONV2D_MIN_CHANNEL_SIZE and num_channels % 32 != 0:
        return ((num_channels + 31) // 32) * 32
    else:
        return num_channels


# --- Mock Classes ---
class MockLayerNorm:
    """Simulates torch.nn.LayerNorm"""
    def __init__(self, hidden_size):
        self.weight = np.random.randn(hidden_size).astype(np.float32)
        self.bias = np.random.randn(hidden_size).astype(np.float32)


class MockConfig:
    """Mock configuration for Segformer Encoder"""
    def __init__(self):
        self.num_encoder_blocks = 4
        self.patch_sizes = [7, 3, 3, 3]
        self.strides = [4, 2, 2, 2]
        self.hidden_sizes = [32, 64, 160, 256]
        self.num_attention_heads = [1, 2, 5, 8]
        self.sr_ratios = [8, 4, 2, 1]
        self.mlp_ratios = [4, 4, 4, 4]
        self.depths = [2, 2, 2, 2]  # Number of layers in each block
        self.reshape_last_stage = True


class MockSegformerEncoder:
    """Mock Segformer Encoder for parameter extraction"""
    def __init__(self, config):
        self.config = config
        
        # Create patch embeddings
        self.patch_embeddings = []
        for i in range(config.num_encoder_blocks):
            in_channels = 3 if i == 0 else config.hidden_sizes[i - 1]
            self.patch_embeddings.append(
                MockOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=in_channels,
                    hidden_size=config.hidden_sizes[i],
                )
            )
        
        # Create transformer blocks
        self.block = []
        for i in range(config.num_encoder_blocks):
            layers = []
            for j in range(config.depths[i]):
                layers.append(
                    MockSegformerLayer(
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        sr_ratio=config.sr_ratios[i],
                    )
                )
            self.block.append(layers)
        
        # Create layer norms
        self.layer_norm = []
        for i in range(config.num_encoder_blocks):
            self.layer_norm.append(MockLayerNorm(config.hidden_sizes[i]))


def create_custom_mesh_preprocessor(device):
    """
    Preprocessor to extract parameters from mock model and move to device.
    Matches TT-Metal structure.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        if isinstance(model, MockSegformerEncoder):
            # Patch embeddings
            parameters["patch_embeddings"] = {}
            for i in range(4):
                # Calculate input channels for this block
                in_channels = 3 if i == 0 else model.config.hidden_sizes[i - 1]
                padded_in_channels = calculate_padded_channels(in_channels)
                
                # Create preprocessor with padded channels
                overlap_path_embedding_preprocess = create_customer_preprocessor_overlap_path(
                    device, padded_in_channels
                )
                parameters["patch_embeddings"][i] = overlap_path_embedding_preprocess(
                    model.patch_embeddings[i], None, None, None
                )
            
            # Transformer blocks
            parameters["block"] = {}
            for i in range(4):
                parameters["block"][i] = {}
                for j in range(model.config.depths[i]):
                    layer_preprocess = create_customer_preprocessor_layer(device)
                    parameters["block"][i][j] = layer_preprocess(
                        model.block[i][j], None, None, None
                    )
            
            # Layer norms
            parameters["layer_norm"] = {}
            for i in range(4):
                parameters["layer_norm"][i] = {
                    "weight": ttnn.to_device(
                        ttnn.as_tensor(model.layer_norm[i].weight), device
                    ),
                    "bias": ttnn.to_device(
                        ttnn.as_tensor(model.layer_norm[i].bias), device
                    ),
                }
        
        return parameters
    
    return preprocessor


def calculate_output_dimensions(height, width, config):
    """Calculate output dimensions after all encoder blocks"""
    h, w = height, width
    for i in range(config.num_encoder_blocks):
        patch_size = config.patch_sizes[i]
        stride = config.strides[i]
        padding = patch_size // 2
        h = (h + 2 * padding - patch_size) // stride + 1
        w = (w + 2 * padding - patch_size) // stride + 1
    return h, w


def tensor_to_numpy(tensor):
    """Safely convert ttnn tensor to numpy array."""
    try:
        result = ttnn.to_torch(tensor)
        if hasattr(result, 'cpu'):
            result = result.cpu()
        if hasattr(result, 'numpy'):
            result = result.numpy()
        if hasattr(result, 'detach'):
            result = result.detach().numpy()
        # If it's already a numpy array or similar
        if isinstance(result, np.ndarray):
            return result
        # Try to convert to numpy
        return np.array(result, dtype=np.float32)
    except Exception as e:
        print(f"       Warning: Could not convert to numpy: {e}")
        return None


# --- Test Cases ---
test_cases = [
    # (batch_size, num_channels, height, width)
    (1, 3, 512, 512),
]


def test_segformer_encoder():
    print("\n" + "=" * 80)
    print("=== Polaris Segformer Encoder Tests ===")
    print("=" * 80)
    
    passed_count = 0
    device = ttnn.open_device(device_id=0)
    print(f"Device opened: {device}\n")
    
    for params in test_cases:
        batch_size, num_channels, height, width = params
        
        test_name = f"Encoder | Batch {batch_size} | Input {height}x{width}x{num_channels}"
        print(f"\n[TEST] {test_name}")
        
        try:
            # 1. Create configuration
            config = MockConfig()
            print(f"       Encoder blocks: {config.num_encoder_blocks}")
            print(f"       Hidden sizes: {config.hidden_sizes}")
            print(f"       Depths: {config.depths}")
            
            # 2. Create mock model
            mock_model = MockSegformerEncoder(config)
            
            # 3. Extract parameters using preprocessor
            preprocessor = create_custom_mesh_preprocessor(device)
            parameters = preprocessor(mock_model, None, None, None)
            
            # 4. Create input tensor in NCHW format and pad channels if needed
            np.random.seed(42)
            input_np = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
            
            # Pad input channels to match the padded weight
            padded_channels = calculate_padded_channels(num_channels)
            if padded_channels > num_channels:
                pad_size = padded_channels - num_channels
                # Pad along the channels dimension (axis=1)
                input_np = np.pad(input_np, ((0, 0), (0, pad_size), (0, 0), (0, 0)), mode='constant')
                print(f"       Padded input channels: {num_channels} → {padded_channels}")
            
            input_tensor = create_polaris_tensor(input_np, device)
            print(f"       Input shape: {input_tensor.shape}")
            
            # 5. Create Polaris model
            ttnn_model = TtSegformerEncoder(config, parameters)
            
            # 6. Run forward pass
            output = ttnn_model(
                device,
                input_tensor,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # 7. Get output
            last_hidden_state = output.last_hidden_state
            
            # Get output shape
            if hasattr(last_hidden_state, 'shape'):
                out_shape = tuple(last_hidden_state.shape)
            else:
                out_shape = tuple(last_hidden_state.data.shape)
            
            print(f"       Output shape: {out_shape}")
            
            # 8. Calculate expected output dimensions
            expected_h, expected_w = calculate_output_dimensions(height, width, config)
            expected_channels = config.hidden_sizes[-1]
            
            # Expected shape depends on reshape_last_stage config
            if config.reshape_last_stage:
                expected_shape = (batch_size, expected_h * expected_w, expected_channels)
            else:
                expected_shape = (batch_size, expected_h, expected_w, expected_channels)
            
            print(f"       Expected shape: {expected_shape}")
            
            # 9. Verify output shape
            shape_match = out_shape == expected_shape
            
            # 10. Verify hidden states if requested
            hidden_states_ok = True
            if output.hidden_states is not None:
                hidden_states_ok = len(output.hidden_states) == config.num_encoder_blocks
                print(f"       Hidden states collected: {len(output.hidden_states)}")
            
            # 11. Try to validate output values (optional - don't fail if conversion doesn't work)
            output_valid = True
            ttnn_output_np = tensor_to_numpy(last_hidden_state)
            if ttnn_output_np is not None:
                if np.any(np.isnan(ttnn_output_np)) or np.any(np.isinf(ttnn_output_np)):
                    output_valid = False
                    print(f"       WARNING: Output contains NaN or Inf values")
                else:
                    mean_val = np.mean(np.abs(ttnn_output_np))
                    print(f"       Output mean absolute value: {mean_val:.6f}")
            else:
                print(f"       Note: Could not validate output values (conversion issue)")
            
            if shape_match and hidden_states_ok:
                print(f"[PASS] {test_name}")
                print(f"       Final dimensions: {expected_h}×{expected_w}×{expected_channels}")
                passed_count += 1
            else:
                print(f"[FAIL] {test_name}")
                if not shape_match:
                    print(f"       Expected shape: {expected_shape}")
                    print(f"       Got shape: {out_shape}")
                if not hidden_states_ok:
                    print(f"       Hidden states mismatch")
                
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
    success = test_segformer_encoder()
    sys.exit(0 if success else 1)