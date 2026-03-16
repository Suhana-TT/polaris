# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import traceback 

# Go up THREE levels from workloads/segformer/tests to reach the repo root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import ttsim tensor operations
import ttsim.front.functional.tensor_op as T

# Import your Polaris Port
from workloads.segformer.tt.segformer_selfoutput import TtsimSegformerSelfOutput

def create_polaris_tensor(numpy_array):
    """Helper to safely wrap a NumPy array into a Polaris graph tensor."""
    # SimTensor expects a 'cfg' dictionary, not a raw array!
    cfg = {
        "name": "dummy_input_tensor",
        "data": numpy_array,
        "shape": list(numpy_array.shape)
    }
    return T.SimTensor(cfg)

def run_tests():
    test_parameters = [
        (1, 16384, 32, 0, 0),
        (1, 16384, 32, 0, 1),
        (1, 4096, 64, 1, 0),
        (1, 4096, 64, 1, 1),
        (1, 1024, 160, 2, 0),
        (1, 1024, 160, 2, 1),
        (1, 256, 256, 3, 0),
        (1, 256, 256, 3, 1),
    ]

    all_passed = True

    print("Starting Polaris Segformer SelfOutput Verification")

    for params in test_parameters:
        batch_size, seq_len, hidden_size, block_i, self_output_i = params
        test_name = f"Block {block_i}, Output {self_output_i} | Shape: ({batch_size}, {seq_len}, {hidden_size})"
        
        try:
            numpy_input = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

            polaris_input_tensor = create_polaris_tensor(numpy_input)

            mock_parameters = {
                "dense": {
                    "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
                    "bias": np.random.randn(hidden_size).astype(np.float32)
                }
            }

            ttsim_model = TtsimSegformerSelfOutput(
                name=f"test_self_output_{block_i}_{self_output_i}",
                hidden_size=hidden_size,
                parameters=mock_parameters
            )
            
            ttsim_output = ttsim_model(polaris_input_tensor)

            expected_shape = (batch_size, seq_len, hidden_size)
            
            out_shape = tuple(ttsim_output.shape) 
            if out_shape != expected_shape:
                raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {out_shape}")

            print(f"[PASSED] {test_name}")

        except Exception as e:
            print(f"[FAILED] {test_name}")
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED. CHECK LOGS ABOVE.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
