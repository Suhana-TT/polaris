# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_model import TtSegformerModel
from workloads.segformer.tests.test_segformer_encoder import (
    create_custom_mesh_preprocessor as create_customer_preprocessor_encoder,
    MockSegformerEncoder,
    calculate_padded_channels,
    calculate_output_dimensions,
)


# --- Mock Classes ---
class MockConfig:
    """Mock configuration for Segformer Model"""
    def __init__(self):
        self.num_encoder_blocks = 4
        self.patch_sizes = [7, 3, 3, 3]
        self.strides = [4, 2, 2, 2]
        self.hidden_sizes = [32, 64, 160, 256]
        self.num_attention_heads = [1, 2, 5, 8]
        self.sr_ratios = [8, 4, 2, 1]
        self.mlp_ratios = [4, 4, 4, 4]
        self.depths = [2, 2, 2, 2]
        self.reshape_last_stage = True
        # Config attributes for model
        self.output_attentions = False
        self.output_hidden_states = True
        self.use_return_dict = True


class MockSegformerModel:
    """Mock Segformer Model for parameter extraction"""
    def __init__(self, config):
        self.config = config
        self.encoder = MockSegformerEncoder(config)


def create_custom_mesh_preprocessor(device):
    """
    Preprocessor to extract parameters from mock model and move to device.
    Matches TT-Metal structure.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        if isinstance(model, MockSegformerModel):
            # Get encoder parameters using the encoder preprocessor
            encoder_preprocessor = create_customer_preprocessor_encoder(device)
            parameters["encoder"] = encoder_preprocessor(model.encoder, None, None, None)
        
        return parameters
    
    return preprocessor


def create_polaris_tensor(numpy_array, device):
    """Convert numpy array to Polaris ttnn.Tensor and move to device."""
    tensor = ttnn.as_tensor(numpy_array.astype(np.float32))
    tensor = ttnn.to_device(tensor, device)
    return tensor


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
        if isinstance(result, np.ndarray):
            return result
        return np.array(result, dtype=np.float32)
    except Exception as e:
        print(f"       Warning: Could not convert to numpy: {e}")
        return None


# --- Test Cases ---
test_cases = [
    # (batch_size, num_channels, height, width)
    (1, 3, 512, 512),
]


def test_segformer_model(device, batch_size, num_channels, height, width):
    """Test Segformer Model - verify it runs and shapes are correct"""
    
    test_name = f"Model | Batch {batch_size} | Input {height}x{width}x{num_channels}"
    print(f"\n[TEST] {test_name}")
    
    try:
        # 1. Create configuration
        config = MockConfig()
        print(f"       Encoder blocks: {config.num_encoder_blocks}")
        print(f"       Hidden sizes: {config.hidden_sizes}")
        print(f"       Depths: {config.depths}")
        
        # 2. Create mock model for parameter extraction
        mock_model = MockSegformerModel(config)
        
        # 3. Extract parameters using preprocessor
        preprocessor = create_custom_mesh_preprocessor(device)
        parameters = preprocessor(mock_model, None, None, None)
        
        # 4. Create input tensor in NCHW format
        np.random.seed(42)
        input_np = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
        
        # Pad input channels to match the minimum channel requirement
        padded_channels = calculate_padded_channels(num_channels)
        if padded_channels > num_channels:
            pad_size = padded_channels - num_channels
            # Pad along the channels dimension (axis=1)
            input_np = np.pad(input_np, ((0, 0), (0, pad_size), (0, 0), (0, 0)), mode='constant')
            print(f"       Padded input channels: {num_channels} → {padded_channels}")
        
        input_tensor = create_polaris_tensor(input_np, device)
        print(f"       Input shape: {input_tensor.shape}")
        
        # 5. Create Polaris model
        ttnn_model = TtSegformerModel(config, parameters)
        
        # 6. Run forward pass
        ttnn_output = ttnn_model(
            device,
            input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        
        # 7. Get output
        last_hidden_state = ttnn_output[0]
        
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
        
        # 10. Verify hidden states if returned
        hidden_states_ok = True
        if hasattr(ttnn_output, 'hidden_states') and ttnn_output.hidden_states is not None:
            hidden_states_ok = len(ttnn_output.hidden_states) == config.num_encoder_blocks
            print(f"       Hidden states collected: {len(ttnn_output.hidden_states)}")
        
        # 11. Optional: validate output values
        ttnn_output_np = tensor_to_numpy(last_hidden_state)
        if ttnn_output_np is not None:
            if np.any(np.isnan(ttnn_output_np)) or np.any(np.isinf(ttnn_output_np)):
                print(f"       WARNING: Output contains NaN or Inf values")
            else:
                mean_val = np.mean(np.abs(ttnn_output_np))
                print(f"       Output mean absolute value: {mean_val:.6f}")
        else:
            print(f"       Note: Could not validate output values (conversion issue)")
        
        if shape_match and hidden_states_ok:
            print(f"[PASS] {test_name}")
            print(f"       Final dimensions: {expected_h}×{expected_w}×{expected_channels}")
            return True
        else:
            print(f"[FAIL] {test_name}")
            if not shape_match:
                print(f"       Expected shape: {expected_shape}")
                print(f"       Got shape: {out_shape}")
            if not hidden_states_ok:
                print(f"       Hidden states mismatch")
            return False
            
    except Exception as e:
        print(f"[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(device):
    """Run all Segformer Model tests"""
    print("\n" + "=" * 80)
    print("=== Polaris Segformer Model Tests ===")
    print("=" * 80)
    
    passed_count = 0
    
    for params in test_cases:
        batch_size, num_channels, height, width = params
        
        if test_segformer_model(device, batch_size, num_channels, height, width):
            passed_count += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{len(test_cases)} passed")
    print("=" * 80)
    
    return passed_count == len(test_cases)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    print(f"Device opened: {device}")
    
    try:
        all_passed = run_all_tests(device)
        sys.exit(0 if all_passed else 1)
    finally:
        ttnn.close_device(device)
        print("Device closed.")