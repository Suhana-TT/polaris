# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import traceback

sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")

import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_dwconv import TtsimSegformerDWConv

def create_polaris_tensor(numpy_array):
    cfg = {
        "name": "dummy_input_tensor",
        "data": numpy_array,
        "shape": list(numpy_array.shape),
        "dtype": np.float32  # FIX: Use numpy type
    }
    return T.SimTensor(cfg)

def run_tests():
    test_parameters = [
        (1, 16384, 128, 128, 128, 0, 0),
        (1, 16384, 128, 128, 128, 0, 1),
        (1, 4096, 256, 64, 64, 1, 0),
        (1, 4096, 256, 64, 64, 1, 1),
        (1, 1024, 640, 32, 32, 2, 0),
        (1, 1024, 640, 32, 32, 2, 1),
        (1, 256, 1024, 16, 16, 3, 0),
        (1, 256, 1024, 16, 16, 3, 1),
    ]

    all_passed = True
    print("=== Starting Polaris Segformer DWConv Verification ===")

    for params in test_parameters:
        batch_size, seq_len, dim, height, width, block_i, dwconv_i = params
        test_name = f"Block {block_i} DWConv {dwconv_i} | In: ({batch_size}, {seq_len}, {dim})"
        
        try:
            numpy_input = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
            polaris_input_tensor = create_polaris_tensor(numpy_input)

            mock_parameters = {
                "dwconv": {
                    "weight": np.random.randn(dim, 1, 3, 3).astype(np.float32),
                    "bias": np.random.randn(dim).astype(np.float32)
                }
            }

            ttsim_model = TtsimSegformerDWConv(
                name=f"test_dwconv_{block_i}_{dwconv_i}",
                parameters=mock_parameters,
                dim=dim
            )
            
            ttsim_output = ttsim_model(polaris_input_tensor, height, width)

            expected_shape = (batch_size, seq_len, dim)
            out_shape = tuple(ttsim_output.shape) 
            if out_shape != expected_shape:
                raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {out_shape}")

            print(f"[PASSED] {test_name}")

        except Exception as e:
            print(f"[FAILED] {test_name}")
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
