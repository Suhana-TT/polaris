# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import math
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_decode_head import TtSegformerDecodeHead
from workloads.segformer.tests.test_segformer_mlp import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_mlp,
    MockSegformerMLP,
)

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
class MockConfig:
    """Mock configuration for Segformer Decode Head"""
    def __init__(self):
        self.num_encoder_blocks = 4
        self.hidden_sizes = [32, 64, 160, 256]
        self.decoder_hidden_size = 256
        self.num_labels = 150

class MockLinear:
    """Simulates torch.nn.Linear"""
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.bias = np.random.randn(out_features).astype(np.float32) * 0.01

class MockConv2d:
    """Simulates torch.nn.Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size):
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.weight = np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * 0.02
        self.bias = np.random.randn(out_channels).astype(np.float32) * 0.01

class MockBatchNorm2d:
    """Simulates torch.nn.BatchNorm2d"""
    def __init__(self, num_features):
        self.weight = np.ones(num_features).astype(np.float32)
        self.bias = np.zeros(num_features).astype(np.float32)
        self.running_mean = np.zeros(num_features).astype(np.float32)
        self.running_var = np.ones(num_features).astype(np.float32)
        self.eps = 1e-5

class MockSegformerDecodeHead:
    """Mock Segformer Decode Head for parameter extraction"""
    def __init__(self, config):
        self.config = config
        
        # MLPs for each encoder block
        self.linear_c = []
        for i in range(config.num_encoder_blocks):
            mlp = MockSegformerMLP(config.hidden_sizes[i], config.decoder_hidden_size)
            self.linear_c.append(mlp)
        
        # Fuse conv and batch norm
        # Input channels = decoder_hidden_size * num_encoder_blocks = 256 * 4 = 1024
        total_channels = config.decoder_hidden_size * config.num_encoder_blocks
        self.linear_fuse = MockConv2d(total_channels, config.decoder_hidden_size, 1)
        self.batch_norm = MockBatchNorm2d(config.decoder_hidden_size)
        
        # Classifier
        self.classifier = MockConv2d(config.decoder_hidden_size, config.num_labels, 1)

def fold_batch_norm2d_into_conv2d(conv, bn):
    """
    Fold batch normalization into convolution weights.
    
    This is a common optimization that combines conv and bn into a single conv.
    new_weight = weight * (gamma / sqrt(var + eps))
    new_bias = gamma * (bias - mean) / sqrt(var + eps) + beta
    """
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    # Compute scale factor
    std = np.sqrt(var + eps)
    scale = gamma / std
    
    # Fold into conv weight: [out_c, in_c, kh, kw] * scale[out_c]
    # Need to broadcast scale across the weight dimensions
    new_weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    
    # Fold into conv bias
    new_bias = gamma * (conv.bias - mean) / std + beta
    
    return new_weight, new_bias

def create_custom_mesh_preprocessor(device):
    """
    Preprocessor to extract parameters from mock decode head and move to device.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        if isinstance(model, MockSegformerDecodeHead):
            # MLP parameters for each encoder block
            parameters["linear_c"] = {}
            for i in range(model.config.num_encoder_blocks):
                mlp_preprocess = create_custom_preprocessor_mlp(device)
                parameters["linear_c"][i] = mlp_preprocess(model.linear_c[i], None, None, None)
            
            # Fold batch norm into linear_fuse conv
            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.linear_fuse, model.batch_norm
            )
            
            # Linear fuse parameters
            # Weight: [out_c, in_c, kh, kw] - keep as is
            # Bias: [C_out] - 1D, NOT reshaped to [1, 1, 1, C]
            parameters["linear_fuse"] = {
                "weight": ttnn.to_device(
                    ttnn.as_tensor(conv_weight), device
                ),
                "bias": ttnn.to_device(
                    ttnn.as_tensor(conv_bias), device  # 1D bias, no reshape!
                ),
            }
            
            # Classifier parameters
            # Weight: [out_c, in_c, kh, kw] - keep as is
            # Bias: [C_out] - 1D, NOT reshaped
            parameters["classifier"] = {
                "weight": ttnn.to_device(
                    ttnn.as_tensor(model.classifier.weight), device
                ),
                "bias": ttnn.to_device(
                    ttnn.as_tensor(model.classifier.bias), device  # 1D bias, no reshape!
                ),
            }
        
        return parameters
    
    return preprocessor

def test_segformer_decode_head(device):
    """Test Segformer Decode Head - verify it runs and shapes are correct"""
    
    test_name = "DecodeHead | Batch 1 | Labels 150"
    print(f"\n[TEST] {test_name}")
    
    try:
        np.random.seed(42)
        batch_size = 1
        
        # Create configuration
        config = MockConfig()
        print(f"       Encoder blocks: {config.num_encoder_blocks}")
        print(f"       Hidden sizes: {config.hidden_sizes}")
        print(f"       Decoder hidden size: {config.decoder_hidden_size}")
        print(f"       Num labels: {config.num_labels}")
        
        # Create input tensors matching encoder outputs
        # Original shapes: [batch, channels, height, width] (NCHW)
        # Folded shapes: [batch, seq_len, channels] (sequence format)
        input_configs = [
            (32, 128, 128),   # Block 0: 32 channels, 128x128 spatial
            (64, 64, 64),     # Block 1: 64 channels, 64x64 spatial
            (160, 32, 32),    # Block 2: 160 channels, 32x32 spatial
            (256, 16, 16),    # Block 3: 256 channels, 16x16 spatial
        ]
        
        print(f"\n       Creating encoder hidden states:")
        ttnn_inputs = []
        for i, (channels, height, width) in enumerate(input_configs):
            seq_len = height * width
            # Create folded tensor [batch, seq_len, channels]
            input_np = np.random.randn(batch_size, seq_len, channels).astype(np.float32)
            ttnn_input = create_polaris_tensor(input_np, device)
            ttnn_inputs.append(ttnn_input)
            print(f"         Block {i}: [{batch_size}, {seq_len}, {channels}] "
                  f"(spatial: {height}x{width})")
        
        ttnn_input_tensor = tuple(ttnn_inputs)
        
        # Create mock model for parameter extraction
        mock_model = MockSegformerDecodeHead(config)
        
        # Extract parameters
        preprocessor = create_custom_mesh_preprocessor(device)
        parameters = preprocessor(mock_model, None, None, None)
        
        # Create Polaris model
        print(f"\n       Creating TtSegformerDecodeHead...")
        ttnn_model = TtSegformerDecodeHead(config, parameters)
        
        # Run forward pass
        print(f"       Running forward pass...")
        ttnn_output = ttnn_model(device, ttnn_input_tensor)
        
        # Get output shape
        if hasattr(ttnn_output, 'shape'):
            actual_shape = tuple(ttnn_output.shape)
        else:
            actual_shape = tuple(ttnn_output.data.shape)
        
        print(f"\n       Output shape: {actual_shape}")
        
        # Expected shape: [batch, num_labels, height, width] in NCHW
        # Target size = 128 (matches first encoder block spatial size)
        target_size = 128
        expected_shape_nhwc = (batch_size, target_size, target_size, config.num_labels)
        expected_shape_nchw = (batch_size, config.num_labels, target_size, target_size)
        
        print(f"       Expected (NHWC): {expected_shape_nhwc}")
        print(f"       Expected (NCHW): {expected_shape_nchw}")
        
        # Check if shape matches either format
        shape_match = actual_shape == expected_shape_nhwc or actual_shape == expected_shape_nchw
        
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
            print(f"\n[PASS] {test_name}")
            print(f"       Output: {actual_shape}")
            return True
        else:
            print(f"\n[FAIL] {test_name}")
            print(f"       Expected shape: {expected_shape_nhwc} or {expected_shape_nchw}")
            print(f"       Got shape: {actual_shape}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False

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