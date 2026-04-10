# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import traceback

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_mlp import TtsimSegformerMLP

def create_polaris_tensor(numpy_array):
    cfg = {
        "name": "dummy_input_tensor",
        "data": numpy_array.astype(np.float32),
        "shape": list(numpy_array.shape),
        "dtype": "float32",
    }
    return T.SimTensor(cfg)

def run_tests():
    # (input_dim, mlp_id, batch_size, height, width)
    test_parameters = [
        (32, 0, 1, 128, 128),
        (64, 1, 1, 64, 64),
        (160, 2, 1, 32, 32),
        (256, 3, 1, 16, 16),
    ]

    decoder_hidden_size = 256
    all_passed = True

    print("==================================================")
    print("Starting Polaris Segformer MLP Verification")
    print("==================================================")

    for params in test_parameters:
        input_dim, mlp_id, batch_size, height, width = params
        test_name = f"MLP ID {mlp_id} | Orig Spatial: ({height}x{width})"
        
        try:
            # 1. GENERATE DUMMY INPUT (Already flattened into 3D: Batch, Seq, Channels)
            # This matches the original ttnn hardware test!
            seq_len = height * width
            numpy_input = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

            # 2. CONVERT TO POLARIS TENSOR
            polaris_input_tensor = create_polaris_tensor(numpy_input)

            # 3. MOCK PARAMETERS
            mock_parameters = {
                "proj": {
                "weight": np.random.randn(decoder_hidden_size, input_dim).astype(np.float32),
                "bias": np.random.randn(decoder_hidden_size).astype(np.float32),
            }
        }

            # 4. INITIALIZE POLARIS MODEL
            ttsim_model = TtsimSegformerMLP(
                name=f"test_mlp_{mlp_id}",
                parameters=mock_parameters
            )
            
            # 5. EXECUTE POLARIS GRAPH
            ttsim_output = ttsim_model(polaris_input_tensor)

            # 6. VERIFY SHAPES
            expected_shape = (batch_size, seq_len, decoder_hidden_size)
            
            out_shape = tuple(ttsim_output.shape) 
            if out_shape != expected_shape:
                raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {out_shape}")

            print(f"[PASSED] {test_name} -> Out: {out_shape}")

        except Exception as e:
            print(f"[FAILED] {test_name}")
            print("--- FULL TRACEBACK ---")
            traceback.print_exc()
            print("----------------------")
            all_passed = False

    print("==================================================")
    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED. CHECK LOGS ABOVE.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
