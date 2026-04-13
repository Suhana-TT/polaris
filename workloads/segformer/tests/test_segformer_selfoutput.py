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
from workloads.segformer.tt.segformer_selfoutput import TtsimSegformerSelfOutput


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


class MockSelfOutput:
    """Simulates SegformerSelfOutput (has .dense)"""
    def __init__(self, hidden_size):
        self.dense = MockLinear(hidden_size, hidden_size)


def create_custom_mesh_preprocessor(device):
    """
    Same structure as TT-Metal create_custom_mesh_preprocessor.
    Extracts parameters from model object, converts to ttnn.Tensor, and moves to device.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        parameters["dense"] = {}
        # Convert numpy arrays to ttnn.Tensor and move to device!
        weight = ttnn.as_tensor(model.dense.weight)
        weight = ttnn.to_device(weight, device)
        parameters["dense"]["weight"] = weight
        
        bias = ttnn.as_tensor(model.dense.bias)
        bias = ttnn.to_device(bias, device)
        parameters["dense"]["bias"] = bias
        return parameters
    return preprocessor


# --- PARAMETER LIST (Same as TT-Metal) ---
test_cases = [
    (1, 16384, 32, 0, 0),
    (1, 16384, 32, 0, 1),
    (1, 4096, 64, 1, 0),
    (1, 4096, 64, 1, 1),
    (1, 1024, 160, 2, 0),
    (1, 1024, 160, 2, 1),
    (1, 256, 256, 3, 0),
    (1, 256, 256, 3, 1),
]


def test_segformer_selfoutput():
    print("=== Starting Polaris Segformer SelfOutput Verification ===")
    all_passed = True

    # Initialize device
    device = ttnn.open_device()
    print(f"Device opened: {device}")

    for params in test_cases:
        batch_size, seq_len, hidden_size, block_i, self_output_i = params
        
        test_name = f"Block {block_i} | Output {self_output_i} | Seq {seq_len} | Dim {hidden_size}"
        
        try:
            mock_model = MockSelfOutput(hidden_size)
            
            preprocessor = create_custom_mesh_preprocessor(device)
            parameters = preprocessor(mock_model, None, None, None)
            
            input_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
            input_tensor = create_polaris_tensor(input_np, device)
            model = TtsimSegformerSelfOutput(
                name=f"selfoutput_{block_i}_{self_output_i}",
                hidden_size=hidden_size,
                parameters=parameters
            )
            
            output = model(input_tensor)
            
            if hasattr(output, 'shape'):
                out_shape = tuple(output.shape)
            elif hasattr(output, 'data'):
                out_shape = tuple(output.data.shape)
            else:
                out_shape = tuple(np.array(output).shape)
            
            expected_shape = (batch_size, seq_len, hidden_size)
            
            if out_shape == expected_shape:
                print(f"[PASSED] {test_name} -> Shape: {out_shape}")
            else:
                print(f"[FAILED] {test_name} -> Expected {expected_shape}, got {out_shape}")
                all_passed = False
                
        except Exception as e:
            print(f"[ERROR]  {test_name} -> {str(e)}")
            traceback.print_exc()
            all_passed = False

    ttnn.close_device(device)
    print("Device closed.")

    print("\n" + "="*50)
    if all_passed:
        print("All SelfOutput configurations passed!")
    else:
        print("Some tests failed.")
    
    return all_passed


if __name__ == "__main__":
    success = test_segformer_selfoutput()
    sys.exit(0 if success else 1)