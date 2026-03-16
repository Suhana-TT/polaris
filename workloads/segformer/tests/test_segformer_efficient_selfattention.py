# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import traceback

# We are forcing Python to recognize the root of your project
sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")

import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_efficient_selfattention import TtsimSegformerEfficientSelfAttention

def create_polaris_tensor(numpy_array):
    cfg = {
        "name": "dummy_input",
        "data": numpy_array,
        "shape": list(numpy_array.shape),
        "dtype": "float32"
    }
    return T.SimTensor(cfg)

def create_mock_parameters(hidden_size, sr_ratio):
    params = {
        "query": {
            "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
            "bias": np.random.randn(hidden_size).astype(np.float32)
        },
        "key": {
            "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
            "bias": np.random.randn(hidden_size).astype(np.float32)
        },
        "value": {
            "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
            "bias": np.random.randn(hidden_size).astype(np.float32)
        },
        "layer_norm": {
            "weight": np.random.randn(hidden_size).astype(np.float32),
            "bias": np.random.randn(hidden_size).astype(np.float32)
        }
    }
    if sr_ratio > 1:
        params["sr"] = {
            "weight": np.random.randn(hidden_size, hidden_size, sr_ratio, sr_ratio).astype(np.float32),
            "bias": np.random.randn(hidden_size).astype(np.float32)
        }
    return params

def run_tests():
    test_parameters = [
        (1, 16384, 32, 128, 128, 1, 8, 0, 0),
        (1, 16384, 32, 128, 128, 1, 8, 0, 1),
        (1, 4096, 64, 64, 64, 2, 4, 1, 0),
        (1, 4096, 64, 64, 64, 2, 4, 1, 1),
        (1, 1024, 160, 32, 32, 5, 2, 2, 0),
        (1, 1024, 160, 32, 32, 5, 2, 2, 1),
        (1, 256, 256, 16, 16, 8, 1, 3, 0),
        (1, 256, 256, 16, 16, 8, 1, 3, 1),
    ]

    all_passed = True
    print("=== Starting Polaris Segformer EfficientSelfAttention Verification ===")

    for params in test_parameters:
        b, s, c, h, w, heads, sr, block_i, attn_i = params
        test_name = f"Block {block_i} | Shape: ({b}, {s}, {c}) | Heads: {heads} | SR: {sr}"
        
        try:
            numpy_input = np.random.randn(b, s, c).astype(np.float32)
            polaris_input = create_polaris_tensor(numpy_input)
            mock_params = create_mock_parameters(c, sr)

            model = TtsimSegformerEfficientSelfAttention(
                name=f"attn_{block_i}_{attn_i}",
                hidden_size=c,
                num_attention_heads=heads,
                parameters=mock_params,
                sequence_reduction_ratio=sr
            )
            
            outputs = model(polaris_input, h, w)
            out_tensor = outputs[0] if isinstance(outputs, tuple) else outputs

            expected_shape = (b, s, c)
            if tuple(out_tensor.shape) != expected_shape:
                raise ValueError(f"Shape mismatch! Expected {expected_shape}, got {tuple(out_tensor.shape)}")

            print(f"[PASSED] {test_name}")

        except Exception as e:
            print(f"[FAILED] {test_name}")
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print(" ALL TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
