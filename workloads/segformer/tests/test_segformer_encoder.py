# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
from typing import Any, Dict, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
import workloads.segformer.tt.segformer_encoder as class_module

class DummyConfig:
    def __init__(self) -> None:
        self.num_encoder_blocks = 4
        self.patch_sizes = [7, 3, 3, 3]
        self.strides = [4, 2, 2, 2]
        self.num_channels = 3
        self.hidden_sizes = [32, 64, 160, 256]
        self.depths = [2, 2, 2, 2]
        self.num_attention_heads = [1, 2, 5, 8]
        self.sr_ratios = [8, 4, 2, 1]
        self.mlp_ratios = [4, 4, 4, 4]
        self.reshape_last_stage = True

# --- MOCK MODULES ---

class DummyPatchEmbeddings:
    """
    TT-Metal encoder passes NHWC into patch embeddings and receives:
      - tokens: [B, S, C]
      - H, W
    """
    def __init__(self, name: str, parameters: Dict[str, Any], stride: int, patch_size: int) -> None:
        self.stride = stride
        self.hidden_size = parameters.get("expected_channels", 32)
        self.name = name

    def __call__(self, pixel_values: Any) -> Tuple[Any, int, int]:
        # Expect NHWC
        B = int(pixel_values.shape[0])
        H = int(pixel_values.shape[1]) // self.stride
        W = int(pixel_values.shape[2]) // self.stride
        S = H * W

        out = T.SimTensor(
            {
                "name": f"{self.name}_dummy_out",
                "data": np.random.randn(B, S, self.hidden_size).astype(np.float32),
                "shape": [B, S, self.hidden_size],
                "dtype": "float32",
            }
        )
        return out, H, W

class DummyLayer:
    """
    TT-Metal encoder blocks operate on 3D [B, S, C] inside the encoder.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, hidden_states: Any, height: int, width: int, output_attentions: bool = False):
        if output_attentions:
            attn = T.SimTensor(
                {
                    "name": "dummy_attn",
                    "data": np.zeros((1,), dtype=np.float32),
                    "shape": [1],
                    "dtype": "float32",
                }
            )
            return (hidden_states, attn)
        return (hidden_states,)

# Monkey-patch before import
setattr(class_module, "TtsimSegformerOverlapPatchEmbeddings", DummyPatchEmbeddings)
setattr(class_module, "TtsimSegformerLayer", DummyLayer)

from workloads.segformer.tt.segformer_encoder import TtsimSegformerEncoder

def run_tests() -> None:
    print("=== Starting Polaris Segformer Encoder Verification ===")
    config = DummyConfig()

    batch_size = 1

    # TT-Metal sends NHWC into encoder
    pixel_values = T.SimTensor(
        {
            "name": "pixel_values",
            "data": np.random.randn(batch_size, 512, 512, 3).astype(np.float32),
            "shape": [batch_size, 512, 512, 3],
            "dtype": "float32",
        }
    )

    params: Dict[str, Any] = {
        "patch_embeddings": [{"expected_channels": c} for c in config.hidden_sizes],
        "block": [[{} for _ in range(d)] for d in config.depths],
        "layer_norm": [
            {
                "weight": np.random.randn(c).astype(np.float32),
                "bias": np.random.randn(c).astype(np.float32),
            }
            for c in config.hidden_sizes
        ],
    }

    try:
        model = TtsimSegformerEncoder("test_encoder", config, params)
        outputs = model(
            pixel_values,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

        print("[PASSED] Segformer Encoder executed successfully!")

        # TT-Metal stores hidden_states before reshape: 3D [B, S, C]
        hidden_states = outputs.hidden_states
        expected_stage_shapes = [
            [1, 128 * 128, 32],   # stage 1
            [1, 64 * 64, 64],     # stage 2
            [1, 32 * 32, 160],    # stage 3
            [1, 16 * 16, 256],    # stage 4
        ]

        print("Stage hidden_states:")
        for i, state in enumerate(hidden_states):
            actual = list(state.shape)
            expected = expected_stage_shapes[i]
            status = "OK" if actual == expected else "MISMATCH"
            print(f"  Stage {i+1}: {actual} | expected {expected} | {status}")

        # TT-Metal final last_hidden_state is NHWC when reshape_last_stage=True
        final_shape = list(outputs.last_hidden_state.shape)
        expected_final_shape = [1, 16, 16, 256]
        status = "OK" if final_shape == expected_final_shape else "MISMATCH"
        print(f"Final last_hidden_state: {final_shape} | expected {expected_final_shape} | {status}")

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()