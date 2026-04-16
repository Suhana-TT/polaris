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
from workloads.segformer.tt.segformer_selfoutput import TtSegformerSelfOutput


def create_polaris_tensor(numpy_array, device):
    """Convert numpy array to Polaris ttnn.Tensor and move to device."""
    tensor = ttnn.as_tensor(numpy_array.astype(np.float32))
    tensor = ttnn.to_device(tensor, device)
    return tensor


# --- Mock Classes ---
class MockLinear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features).astype(np.float32)
        self.bias = np.random.randn(out_features).astype(np.float32)


class MockSelfOutput:
    def __init__(self, hidden_size):
        self.dense = MockLinear(hidden_size, hidden_size)


def create_custom_mesh_preprocessor(device):
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        parameters["dense"] = {}
        parameters["dense"]["weight"] = ttnn.to_device(ttnn.as_tensor(model.dense.weight), device)
        parameters["dense"]["bias"] = ttnn.to_device(ttnn.as_tensor(model.dense.bias), device)
        return parameters
    return preprocessor


# --- Test Cases ---
TEST_CASES = [
    (1, 16384, 32, 0, 0),
    (1, 16384, 32, 0, 1),
    (1, 4096, 64, 1, 0),
    (1, 4096, 64, 1, 1),
    (1, 1024, 160, 2, 0),
    (1, 1024, 160, 2, 1),
    (1, 256, 256, 3, 0),
    (1, 256, 256, 3, 1),
]


def run_all_tests(device, cfg=None, **kwargs):
    """
    Run all Segformer SelfOutput tests.
    
    Args:
        device: Device from Polaris
        cfg: Optional config dict
        **kwargs: Additional Polaris arguments
    """
    print("\n" + "=" * 80)
    print("=== Polaris Segformer SelfOutput Tests ===")
    print("=" * 80)
    
    if cfg:
        print(f"Config: {cfg}")
    
    passed_count = 0
    total_tests = len(TEST_CASES)
    
    for batch_size, seq_len, hidden_size, block_i, self_output_i in TEST_CASES:
        test_name = f"Block {block_i} | Output {self_output_i} | Seq {seq_len} | Dim {hidden_size}"
        
        try:
            # Create mock model
            mock_model = MockSelfOutput(hidden_size)
            
            # Preprocess parameters
            preprocessor = create_custom_mesh_preprocessor(device)
            parameters = preprocessor(mock_model, None, None, None)
            
            # Create input tensor
            input_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
            input_tensor = create_polaris_tensor(input_np, device)
            
            # Create model WITH parameters (your class requires them in __init__)
            model = TtSegformerSelfOutput(
                name=f"selfoutput_{block_i}_{self_output_i}",
                hidden_size=hidden_size,
                parameters=parameters
            )
            
            # Forward pass - just input tensor (parameters already in model)
            output = model(input_tensor)
            
            # Check output shape
            if hasattr(output, 'shape'):
                out_shape = tuple(output.shape)
            elif hasattr(output, 'data'):
                out_shape = tuple(output.data.shape)
            else:
                out_shape = tuple(np.array(output).shape)
            
            expected_shape = (batch_size, seq_len, hidden_size)
            
            if out_shape == expected_shape:
                print(f"[PASSED] {test_name} -> Shape: {out_shape}")
                passed_count += 1
            else:
                print(f"[FAILED] {test_name} -> Expected {expected_shape}, got {out_shape}")
                
        except Exception as e:
            print(f"[ERROR]  {test_name} -> {str(e)}")
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{total_tests} passed")
    print("=" * 80)
    
    return passed_count == total_tests


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    print(f"Device opened: {device}")
    
    try:
        success = run_all_tests(device)
    finally:
        ttnn.close_device(device)
        print("Device closed.")
    
    sys.exit(0 if success else 1)