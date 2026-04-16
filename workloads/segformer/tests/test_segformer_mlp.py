# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_mlp import TtSegformerMLP


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


# --- Mock Classes ---
class MockLinear:
    """Simulates torch.nn.Linear"""
    def __init__(self, in_features, out_features):
        # PyTorch Linear stores weight as [out_features, in_features]
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.bias = np.random.randn(out_features).astype(np.float32) * 0.01


class MockSegformerMLP:
    """Mock Segformer MLP for parameter extraction"""
    def __init__(self, input_dim, output_dim=256):
        self.proj = MockLinear(input_dim, output_dim)


class MockConfig:
    """Mock configuration"""
    def __init__(self):
        self.decoder_hidden_size = 256


def create_custom_mesh_preprocessor(device):
    """
    Preprocessor to extract parameters from mock MLP and move to device.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        if isinstance(model, MockSegformerMLP):
            # Weight shape: [out_features, in_features]
            # TtSegformerMLP will transpose internally
            parameters["proj"] = {
                "weight": ttnn.to_device(
                    ttnn.as_tensor(model.proj.weight), device
                ),
                "bias": ttnn.to_device(
                    ttnn.as_tensor(model.proj.bias), device
                ),
            }
        
        return parameters
    
    return preprocessor


def test_segformer_mlp(device, input_dim, mlp_id, batch_size, height, width):
    """Test Segformer MLP module - verify it runs and shapes are correct"""
    
    test_name = f"MLP {mlp_id} | input_dim={input_dim} | spatial={height}x{width}"
    print(f"\n[TEST] {test_name}")
    
    try:
        np.random.seed(42)
        
        # Create configuration
        config = MockConfig()
        output_dim = config.decoder_hidden_size  # 256
        
        print(f"       Input dim: {input_dim}")
        print(f"       Output dim: {output_dim}")
        print(f"       Batch size: {batch_size}")
        print(f"       Spatial: {height}x{width}")
        
        # Create input tensor
        # Original shape: [batch, input_dim, height, width] (NCHW)
        # Folded shape: [batch, height*width, input_dim] (sequence format)
        seq_len = height * width
        input_np = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        
        ttnn_input = create_polaris_tensor(input_np, device)
        print(f"       Input shape: {ttnn_input.shape}")
        
        # Create mock model for parameter extraction
        mock_model = MockSegformerMLP(input_dim, output_dim)
        
        # Extract parameters
        preprocessor = create_custom_mesh_preprocessor(device)
        parameters = preprocessor(mock_model, None, None, None)
        
        # Create Polaris model
        ttnn_model = TtSegformerMLP(
            parameters=parameters,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        
        # Run forward pass
        ttnn_output = ttnn_model(device, ttnn_input)
        
        # Get output shape
        if hasattr(ttnn_output, 'shape'):
            actual_shape = tuple(ttnn_output.shape)
        else:
            actual_shape = tuple(ttnn_output.data.shape)
        
        print(f"       Output shape: {actual_shape}")
        
        # Expected shape: [batch, seq_len, output_dim]
        expected_shape = (batch_size, seq_len, output_dim)
        print(f"       Expected shape: {expected_shape}")
        
        # Verify shape
        shape_match = actual_shape == expected_shape
        
        # Optional: validate output values
        output_np = tensor_to_numpy(ttnn_output)
        if output_np is not None:
            if np.any(np.isnan(output_np)) or np.any(np.isinf(output_np)):
                print(f"       WARNING: Output contains NaN or Inf values")
            else:
                mean_val = np.mean(np.abs(output_np))
                print(f"       Output mean absolute value: {mean_val:.6f}")
        else:
            print(f"       Note: Could not validate output values")
        
        if shape_match:
            print(f"[PASS] {test_name}")
            return True
        else:
            print(f"[FAIL] {test_name}")
            print(f"       Expected shape: {expected_shape}")
            print(f"       Got shape: {actual_shape}")
            return False
            
    except Exception as e:
        print(f"[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# --- Test Cases ---
# (input_dim, mlp_id, batch_size, height, width)
test_cases = [
    (32, 0, 1, 128, 128),   # Block 0: 32 channels, 128x128 spatial
    (64, 1, 1, 64, 64),     # Block 1: 64 channels, 64x64 spatial
    (160, 2, 1, 32, 32),    # Block 2: 160 channels, 32x32 spatial
    (256, 3, 1, 16, 16),    # Block 3: 256 channels, 16x16 spatial
]


def run_all_tests(device, cfg=None, **kwargs):
    """Run all Segformer Decode Head tests."""
    print("\n" + "=" * 80)
    print("=== Polaris Segformer Decode Head Tests ===")
    print("=" * 80)
    
    if cfg:
        bs = cfg.get('bs', 1)
        num_labels = cfg.get('num_labels', 150)
        print(f"Config: bs={bs}, num_labels={num_labels}")
    
    passed_count = 0
    total_tests = 1
    
    if test_segformer_decode_head(device):
        passed_count += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{total_tests} passed")
    print("=" * 80)
    
    return passed_count == total_tests


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    print(f"Device opened: {device}")
    
    try:
        all_passed = run_all_tests(device, cfg=None)
        sys.exit(0 if all_passed else 1)
    finally:
        ttnn.close_device(device)
        print("Device closed.")