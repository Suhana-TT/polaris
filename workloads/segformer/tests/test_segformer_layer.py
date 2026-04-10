# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import numpy as np
import traceback
from typing import Any, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
import workloads.segformer.tt.segformer_layer as class_module


class DummyConfig:
    def __init__(self) -> None:
        pass


# --- MOCK SUBMODULES ---
class DummyAttention:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, hidden_states: Any, height: int, width: int, output_attentions: bool = False) -> Tuple[Any, None]:
        return (hidden_states, None)


class DummyMixFFN:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, hidden_states: Any, height: int, width: int) -> Any:
        return hidden_states

setattr(class_module, 'TtsimSegformerAttention', DummyAttention)
setattr(class_module, 'TtsimSegformerMixFFN', DummyMixFFN)

# Import after patching
from workloads.segformer.tt.segformer_layer import TtsimSegformerLayer


# --- YOUR PARAMETER LIST ---
# batch_size, seq_len, hidden_size, height, width, num_attention_heads, drop_path, sequence_reduction_ratio, mlp_ratio, block_i, segformer_i
test_cases = [
    (1, 16384, 32, 128, 128, 1, 0, 8, 4, 0, 0),
    (1, 16384, 32, 128, 128, 1, 0, 8, 4, 0, 1),
    (1, 4096, 64, 64, 64, 2, 0, 4, 4, 1, 0),
    (1, 4096, 64, 64, 64, 2, 0, 4, 4, 1, 1),
    (1, 1024, 160, 32, 32, 5, 0, 2, 4, 2, 0),
    (1, 1024, 160, 32, 32, 5, 0, 2, 4, 2, 1),
    (1, 256, 256, 16, 16, 8, 0, 1, 4, 3, 0),
    (1, 256, 256, 16, 16, 8, 0, 1, 4, 3, 1),
]


def run_tests() -> None:
    print("=== Starting Polaris Segformer Layer Verification ===")
    config = DummyConfig()
    all_passed = True

    for case in test_cases:
        batch_size, seq_len, hidden_size, height, width, num_heads, drop_path, sr_ratio, mlp_ratio, block_i, segformer_i = case
        test_name = f"Block {block_i} | Layer {segformer_i} | Seq {seq_len} | Dim {hidden_size}"

        # 1. Mock Input Sequence
        hidden_states = T.SimTensor({
            "name": f"hidden_states_{block_i}_{segformer_i}",
            "data": np.random.randn(batch_size, 1, seq_len, hidden_size).astype(np.float32),
            "shape": [batch_size, 1, seq_len, hidden_size],
            "dtype": "float32"
        })

        # 2. Mock Parameters dynamically based on hidden_size
        params = {
            "layer_norm_1": {
                "weight": np.random.randn(hidden_size).astype(np.float32),
                "bias": np.random.randn(hidden_size).astype(np.float32)
            },
            "layer_norm_2": {
                "weight": np.random.randn(hidden_size).astype(np.float32),
                "bias": np.random.randn(hidden_size).astype(np.float32)
            },
            "attention": {},
            "mlp": {}        
        }

        # 3. Initialize and Run
        try:
            model = TtsimSegformerLayer(
                name=f"test_layer_{block_i}_{segformer_i}",
                config=config,
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                sequence_reduction_ratio=sr_ratio,
                parameters=params,
                mlp_ratio=mlp_ratio
            )
            outputs = model(hidden_states, height, width)
            layer_output = outputs[0]

            expected_shape = [batch_size, 1, seq_len, hidden_size]
            if layer_output.shape == expected_shape:
                print(f"[PASSED] {test_name} -> Output Shape: {layer_output.shape}")
            else:
                print(f"[FAILED] {test_name} -> Expected {expected_shape}, got {layer_output.shape}")
                all_passed = False
        except Exception as e:
            print(f"[ERROR]  {test_name}")
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\nAll Segformer Layer configurations passed successfully!")


if __name__ == "__main__":
    run_tests()