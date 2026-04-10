# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
from typing import Dict, Any

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_decode_head import TtsimSegformerDecodeHead

class DummyConfig:
    def __init__(self) -> None:
        self.num_encoder_blocks = 4
        self.decoder_hidden_size = 256
        self.num_labels = 150

def run_tests() -> None:
    print("=== Polaris Segformer DecodeHead Test ===")

    np.random.seed(42)
    config = DummyConfig()

    # Folded encoder hidden states: [B, S, C]
    shapes = [
        (1, 16384, 32),
        (1, 4096, 64),
        (1, 1024, 160),
        (1, 256, 256),
    ]

    inputs = []
    for i, (b, s, c) in enumerate(shapes):
        inp = T.SimTensor(
            {
                "name": f"input_{i}",
                "data": np.random.randn(b, s, c).astype(np.float32),
                "shape": [b, s, c],
                "dtype": "float32",
            }
        )
        inputs.append(inp)

    params: Dict[str, Any] = {"linear_c": {}, "linear_fuse": {}, "classifier": {}}

    for i in range(4):
        in_c = shapes[i][2]
        out_c = config.decoder_hidden_size
        params["linear_c"][i] = {
            "proj": {
                # Match current Polaris MLP expectation
                "weight": np.random.randn(out_c, in_c).astype(np.float32),
                "bias": np.random.randn(out_c).astype(np.float32),
            }
        }

    fuse_in = config.decoder_hidden_size * 4  # 1024
    params["linear_fuse"] = {
        "weight": np.random.randn(config.decoder_hidden_size, fuse_in, 1, 1).astype(np.float32),
        "bias": np.random.randn(config.decoder_hidden_size).astype(np.float32),
    }

    params["classifier"] = {
        "weight": np.random.randn(config.num_labels, config.decoder_hidden_size, 1, 1).astype(np.float32),
        "bias": np.random.randn(config.num_labels).astype(np.float32),
    }

    model = TtsimSegformerDecodeHead("test_decode_head", config, params)

    try:
        out = model(tuple(inputs))
        expected_shape = [1, 150, 128, 128]  # NCHW for current Polaris path
        actual_shape = list(out.shape)
        match = actual_shape == expected_shape

        print(f"Output shape: {actual_shape} | Expected: {expected_shape} | Match: {match}")
        print("[PASSED]" if match else "[FAILED]")

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tests()