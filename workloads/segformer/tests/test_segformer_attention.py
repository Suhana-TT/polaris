# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import traceback

sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")

import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_attention import TtsimSegformerAttention

def create_polaris_tensor(numpy_array):
    return T.SimTensor({
        "name": "input", 
        "data": numpy_array.astype(np.float32), 
        "shape": list(numpy_array.shape), 
        "dtype": "float32"
    })

def mock_params(hidden_size, sr):
    p = {
        "self": {
            "query": {"weight": np.random.randn(hidden_size, hidden_size).astype(np.float32), "bias": np.random.randn(hidden_size).astype(np.float32)},
            "key": {"weight": np.random.randn(hidden_size, hidden_size).astype(np.float32), "bias": np.random.randn(hidden_size).astype(np.float32)},
            "value": {"weight": np.random.randn(hidden_size, hidden_size).astype(np.float32), "bias": np.random.randn(hidden_size).astype(np.float32)},
            "layer_norm": {"weight": np.random.randn(hidden_size).astype(np.float32), "bias": np.random.randn(hidden_size).astype(np.float32)},
        },
        "output": {
            "dense": {"weight": np.random.randn(hidden_size, hidden_size).astype(np.float32), "bias": np.random.randn(hidden_size).astype(np.float32)},
            "layer_norm": {"weight": np.random.randn(hidden_size).astype(np.float32), "bias": np.random.randn(hidden_size).astype(np.float32)},
        }
    }
    if sr > 1:
        p["self"]["sr"] = {"weight": np.random.randn(hidden_size, hidden_size, sr, sr).astype(np.float32), "bias": np.random.randn(hidden_size).astype(np.float32)}
    return p

# --- YOUR PARAMETER LIST ---
# (hidden_size, num_attention_heads, sequence_reduction_ratio, batch_size, seq_len, height, width, block_i, attention_i)
test_cases = [
    (32, 1, 8, 1, 16384, 128, 128, 0, 0),
    (32, 1, 8, 1, 16384, 128, 128, 0, 1),
    (64, 2, 4, 1, 4096, 64, 64, 1, 0),
    (64, 2, 4, 1, 4096, 64, 64, 1, 1),
    (160, 5, 2, 1, 1024, 32, 32, 2, 0),
    (160, 5, 2, 1, 1024, 32, 32, 2, 1),
    (256, 8, 1, 1, 256, 16, 16, 3, 0),
    (256, 8, 1, 1, 256, 16, 16, 3, 1),
]

class DummyConfig:
    pass

def run_tests():
    print("=== Starting Polaris Attention Verification ===")
    all_passed = True
    config = DummyConfig()

    for params in test_cases:
        # Unpack the list exactly like pytest does
        hidden_size, num_attention_heads, sequence_reduction_ratio, batch_size, seq_len, height, width, block_i, attention_i = params
        
        test_name = f"Block {block_i} | Attn {attention_i} | Seq {seq_len} | Dim {hidden_size}"
        
        try:
            # 1. Generate Input Tensor
            x = create_polaris_tensor(np.random.randn(batch_size, seq_len, hidden_size))
            
            # 2. Initialize Model
            model = TtsimSegformerAttention(
                name=f"attn_{block_i}_{attention_i}",
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                parameters=mock_params(hidden_size, sequence_reduction_ratio),
                sequence_reduction_ratio=sequence_reduction_ratio
            )
            
            # 3. Run Math
            out = model(x, height, width)[0]
            
            # 4. Verify Shape
            expected_shape = [batch_size, seq_len, hidden_size]
            if out.shape == expected_shape:
                print(f"[PASSED] {test_name} -> Output Shape: {out.shape}")
            else:
                print(f"[FAILED] {test_name} -> Expected {expected_shape}, got {out.shape}")
                all_passed = False
                
        except Exception as e:
            print(f"[ERROR]  {test_name}")
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\n All Attention configurations passed successfully!")

if __name__ == "__main__":
    run_tests()