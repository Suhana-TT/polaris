# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_attention import TtsimSegformerAttention

# Import preprocessors from other test files
from workloads.segformer.tests.test_segformer_efficient_selfattention import (
    create_custom_mesh_preprocessor as create_preprocessor_selfattention,
)
from workloads.segformer.tests.test_segformer_selfoutput import (
    create_custom_mesh_preprocessor as create_preprocessor_selfoutput,
)


def create_polaris_tensor(numpy_array, device):
    """Convert numpy array to Polaris ttnn.Tensor and move to device."""
    tensor = ttnn.as_tensor(numpy_array.astype(np.float32))
    tensor = ttnn.to_device(tensor, device)
    return tensor


# --- Mock Classes (Replace PyTorch models) ---

class MockLinear:
    """Simulates torch.nn.Linear"""
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features).astype(np.float32)
        self.bias = np.random.randn(out_features).astype(np.float32)


class MockLayerNorm:
    """Simulates torch.nn.LayerNorm"""
    def __init__(self, hidden_size):
        self.weight = np.random.randn(hidden_size).astype(np.float32)
        self.bias = np.random.randn(hidden_size).astype(np.float32)


class MockConv2d:
    """Simulates torch.nn.Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size):
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)
        self.bias = np.random.randn(out_channels).astype(np.float32)


class MockSelfAttention:
    """Simulates SegformerEfficientSelfAttention (model.self)"""
    def __init__(self, hidden_size, sr):
        self.query = MockLinear(hidden_size, hidden_size)
        self.key = MockLinear(hidden_size, hidden_size)
        self.value = MockLinear(hidden_size, hidden_size)
        self.layer_norm = MockLayerNorm(hidden_size)
        if sr > 1:
            self.sr = MockConv2d(hidden_size, hidden_size, sr)


class MockSelfOutput:
    """Simulates SegformerSelfOutput (model.output)"""
    def __init__(self, hidden_size):
        self.dense = MockLinear(hidden_size, hidden_size)


class MockSegformerAttention:
    """Simulates SegformerAttention (has .self and .output)"""
    def __init__(self, hidden_size, sr):
        self.self = MockSelfAttention(hidden_size, sr)
        self.output = MockSelfOutput(hidden_size)


def create_custom_mesh_preprocessor(device):
    """
    Same structure as TT-Metal create_custom_mesh_preprocessor.
    Extracts parameters from model and moves to device.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        # Get self-attention parameters
        self_attention_preprocessor = create_preprocessor_selfattention(device)
        parameters["self"] = self_attention_preprocessor(model.self, None, None, None)
        
        # Get self-output parameters
        self_output_preprocessor = create_preprocessor_selfoutput(device)
        parameters["output"] = self_output_preprocessor(model.output, None, None, None)
        
        return parameters
    
    return preprocessor


# --- PARAMETER LIST (Same as TT-Metal) ---
test_cases = [
    # (hidden_size, num_attention_heads, sequence_reduction_ratio, batch_size, seq_len, height, width, block_i, attention_i)
    (32, 1, 8, 1, 16384, 128, 128, 0, 0),
    (32, 1, 8, 1, 16384, 128, 128, 0, 1),
    (64, 2, 4, 1, 4096, 64, 64, 1, 0),
    (64, 2, 4, 1, 4096, 64, 64, 1, 1),
    (160, 5, 2, 1, 1024, 32, 32, 2, 0),
    (160, 5, 2, 1, 1024, 32, 32, 2, 1),
    (256, 8, 1, 1, 256, 16, 16, 3, 0),
    (256, 8, 1, 1, 256, 16, 16, 3, 1),
]


def test_segformer_attention():
    print("=== Starting Polaris Segformer Attention Verification ===")
    all_passed = True

    # 1. Open device
    device = ttnn.open_device()
    print(f"Device opened: {device}")

    for params in test_cases:
        hidden_size, num_attention_heads, sequence_reduction_ratio, batch_size, seq_len, height, width, block_i, attention_i = params
        
        test_name = f"Block {block_i} | Attn {attention_i} | Seq {seq_len} | Dim {hidden_size}"
        
        try:
            # 2. Create mock model (replaces PyTorch SegformerAttention)
            mock_model = MockSegformerAttention(hidden_size, sequence_reduction_ratio)
            
            # 3. Use preprocessor to extract parameters (converts to ttnn.Tensor and moves to device)
            preprocessor = create_custom_mesh_preprocessor(device)
            parameters = preprocessor(mock_model, None, None, None)
            
            # 4. Create input tensor and move to device
            input_np = np.random.randn(batch_size, 1, seq_len, hidden_size).astype(np.float32)
            input_tensor = create_polaris_tensor(input_np, device)
            
            # 5. Initialize Polaris model
            model = TtsimSegformerAttention(
                name=f"attention_{block_i}_{attention_i}",
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                parameters=parameters,
                sequence_reduction_ratio=sequence_reduction_ratio,
            )
            
            # 6. Run forward pass
            outputs = model(input_tensor, height, width, output_attentions=False)
            
            # 7. Get output (first element of tuple)
            output = outputs[0]
            
            # 8. Get output shape
            if hasattr(output, 'shape'):
                out_shape = tuple(output.shape)
            elif hasattr(output, 'data'):
                out_shape = tuple(output.data.shape)
            else:
                out_shape = tuple(np.array(output).shape)
            
            # 9. Verify shape
            # Output can be (batch, 1, seq_len, hidden) or (batch, seq_len, hidden)
            expected_shape_4d = (batch_size, 1, seq_len, hidden_size)
            expected_shape_3d = (batch_size, seq_len, hidden_size)
            
            if out_shape == expected_shape_4d or out_shape == expected_shape_3d:
                print(f"[PASSED] {test_name} -> Shape: {out_shape}")
            else:
                print(f"[FAILED] {test_name} -> Expected {expected_shape_4d} or {expected_shape_3d}, got {out_shape}")
                all_passed = False
                
        except Exception as e:
            print(f"[ERROR]  {test_name} -> {str(e)}")
            traceback.print_exc()
            all_passed = False

    # 10. Close device
    ttnn.close_device(device)
    print("Device closed.")

    print("\n" + "="*50)
    if all_passed:
        print("✓ All Attention configurations passed!")
    else:
        print("✗ Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    success = test_segformer_attention()
    sys.exit(0 if success else 1)