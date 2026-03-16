# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import traceback

# Force pathing
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_mix_ffn import TtsimSegformerMixFFN

def create_polaris_tensor(numpy_array, name="input"):
    return T.SimTensor({
        "name": name, 
        "data": numpy_array, 
        "shape": list(numpy_array.shape),
        "dtype": "float32"
    })

# --- THE FULL PARAMETER LIST ---
# in_features, hidden_features, out_features, batch_size, seq_len, height, width, block_i, mixffn_i
test_configs = [
    (32, 128, 32, 1, 16384, 128, 128, 0, 0),
    (32, 128, 32, 1, 16384, 128, 128, 0, 1),
    (64, 256, 64, 1, 4096, 64, 64, 1, 0),
    (64, 256, 64, 1, 4096, 64, 64, 1, 1),
    (160, 640, 160, 1, 1024, 32, 32, 2, 0),
    (160, 640, 160, 1, 1024, 32, 32, 2, 1),
    (256, 1024, 256, 1, 256, 16, 16, 3, 0),
    (256, 1024, 256, 1, 256, 16, 16, 3, 1),
]

def run_mixffn_tests():
    print("=== Starting Polaris Segformer MixFFN Verification ===")
    all_passed = True

    for in_f, hidden_f, out_f, b, s, h, w, block_i, mixffn_i in test_configs:
        test_name = f"Block {block_i} | MixFFN {mixffn_i} | Seq {s} | Dim {in_f}"
        
        try:
            # 1. Mock Input
            np_input = np.random.randn(b, s, in_f).astype(np.float32)
            polaris_input = create_polaris_tensor(np_input, name=f"input_{block_i}_{mixffn_i}")

            # 2. Mock Parameters
            mock_params = {
                "dense1": {
                    "weight": np.random.randn(in_f, hidden_f).astype(np.float32),
                    "bias": np.random.randn(hidden_f).astype(np.float32)
                },
                "dwconv": {
                    "weight": np.random.randn(hidden_f, 1, 3, 3).astype(np.float32),
                    "bias": np.random.randn(hidden_f).astype(np.float32)
                },
                "dense2": {
                    "weight": np.random.randn(hidden_f, out_f).astype(np.float32),
                    "bias": np.random.randn(out_f).astype(np.float32)
                }
            }

            # 3. Initialize Model
            model = TtsimSegformerMixFFN(
                name=f"test_mixffn_{block_i}_{mixffn_i}",
                config=None,
                in_features=in_f,
                hidden_features=hidden_f,
                out_features=out_f,
                parameters=mock_params
            )

            # 4. Run Math & Verify
            output = model(polaris_input, h, w)
            
            expected_shape = [b, s, out_f]
            if output.shape == expected_shape:
                print(f"[PASSED] {test_name} -> Output Shape: {output.shape}")
            else:
                print(f"[FAILED] {test_name} -> Expected {expected_shape}, got {output.shape}")
                all_passed = False

        except Exception:
            print(f"[ERROR]  {test_name}")
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\n All MixFFN configurations passed successfully!")

if __name__ == "__main__":
    run_mixffn_tests()