# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import numpy as np
from typing import Any, Dict, List, Tuple

# Force pathing
sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")
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
    def __init__(self, name: str, parameters: Dict[str, Any], stride: int, patch_size: int) -> None:
        self.stride = stride
        self.hidden_size = parameters.get("expected_channels", 32)
        self.name = name

    def __call__(self, pixel_values: Any) -> Tuple[Any, int, int]:
        # Simulate spatial reduction
        B = int(pixel_values.shape[0])
        H = int(pixel_values.shape[2]) // self.stride
        W = int(pixel_values.shape[3]) // self.stride
        S = H * W
        
        out = T.SimTensor({
            "name": f"{self.name}_dummy_out",
            "data": np.random.randn(B, S, self.hidden_size).astype(np.float32),
            "shape": [B, S, self.hidden_size],
            "dtype": "float32"
        })
        return out, H, W


class DummyLayer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, hidden_states: Any, height: int, width: int, output_attentions: bool = False) -> Tuple[Any]:
        return (hidden_states,)


# Monkey-patch using setattr
setattr(class_module, 'TtsimSegformerOverlapPatchEmbeddings', DummyPatchEmbeddings)
setattr(class_module, 'TtsimSegformerLayer', DummyLayer)

# Import after patching
from workloads.segformer.tt.segformer_encoder import TtsimSegformerEncoder


def run_tests() -> None:
    print("=== Starting Polaris Segformer Encoder Verification ===")
    config = DummyConfig()
    
    # 1. Mock Input Image (512x512)
    batch_size = 1
    pixel_values = T.SimTensor({
        "name": "pixel_values",
        "data": np.random.randn(batch_size, 3, 512, 512).astype(np.float32),
        "shape": [batch_size, 3, 512, 512],
        "dtype": "float32"
    })

    # 2. Mock Parameters
    params: Dict[str, Any] = {
        "patch_embeddings": [{"expected_channels": c} for c in config.hidden_sizes],
        "block": [[{} for _ in range(d)] for d in config.depths],
        "layer_norm": [{"weight": np.random.randn(c).astype(np.float32), "bias": np.random.randn(c).astype(np.float32)} for c in config.hidden_sizes]
    }

    # 3. Initialize and Run
    try:
        model = TtsimSegformerEncoder("test_encoder", config, params)
        outputs = model(pixel_values, output_hidden_states=True, return_dict=True)
        
        print(f"[PASSED] Segformer Encoder executed successfully!")
        print("         Generated 4 Hierarchical Feature Maps:")
        
        hidden_states = outputs.hidden_states  # type: ignore
        for i, state in enumerate(hidden_states):
            print(f"           Stage {i+1}: {state.shape}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_tests()