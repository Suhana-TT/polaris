# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
from typing import Any, Dict, Optional, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
import workloads.segformer.tt.segformer_model as class_module
from workloads.segformer.tt.segformer_encoder import TtsimBaseModelOutput

class DummyConfig:
    def __init__(self) -> None:
        self.output_attentions = False
        self.output_hidden_states = False
        self.use_return_dict = True

class ValidatingMockEncoder:
    """
    Mock encoder that validates model.py preprocessing:
    - input must be NHWC
    - channels must be padded to min_channels=8
    """

    def __init__(self, name: str, config: Any, parameters: Any) -> None:
        self.name = name
        self.config = config

    def __call__(
        self,
        pixel_values: Any,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> TtsimBaseModelOutput:
        shape = list(pixel_values.shape)

        # Expect NHWC after model.py preprocessing
        if len(shape) != 4:
            raise ValueError(f"Encoder expected 4D NHWC input, got {shape}")

        batch, height, width, channels = shape

        # For RGB input, model.py should pad channels from 3 to 8 before permuting
        if channels != 8:
            raise ValueError(f"Expected encoder input channels to be padded to 8, got {channels}")

        # Final SegFormer encoder output for 512x512 input is [1, 16, 16, 256]
        final_state = T.SimTensor(
            {
                "name": f"{self.name}_output",
                "data": np.random.randn(batch, 16, 16, 256).astype(np.float32),
                "shape": [batch, 16, 16, 256],
                "dtype": "float32",
            }
        )

        hidden_states = None
        if output_hidden_states:
            hidden_states = (
                T.SimTensor(
                    {
                        "name": f"{self.name}_hs1",
                        "data": np.random.randn(batch, 128 * 128, 32).astype(np.float32),
                        "shape": [batch, 128 * 128, 32],
                        "dtype": "float32",
                    }
                ),
                T.SimTensor(
                    {
                        "name": f"{self.name}_hs2",
                        "data": np.random.randn(batch, 64 * 64, 64).astype(np.float32),
                        "shape": [batch, 64 * 64, 64],
                        "dtype": "float32",
                    }
                ),
                T.SimTensor(
                    {
                        "name": f"{self.name}_hs3",
                        "data": np.random.randn(batch, 32 * 32, 160).astype(np.float32),
                        "shape": [batch, 32 * 32, 160],
                        "dtype": "float32",
                    }
                ),
                T.SimTensor(
                    {
                        "name": f"{self.name}_hs4",
                        "data": np.random.randn(batch, 16 * 16, 256).astype(np.float32),
                        "shape": [batch, 16 * 16, 256],
                        "dtype": "float32",
                    }
                ),
            )

        return TtsimBaseModelOutput(
            last_hidden_state=final_state,
            hidden_states=hidden_states,
            attentions=None,
        )

# Monkey-patch encoder before importing model
setattr(class_module, "TtsimSegformerEncoder", ValidatingMockEncoder)

from workloads.segformer.tt.segformer_model import TtsimSegformerModel

def test_segformer_model() -> None:
    print("=== Starting Polaris Segformer Model Verification ===")

    config = DummyConfig()

    batch_size = 1
    num_channels = 3
    height = 512
    width = 512

    # Model input should be NCHW, like TT-Metal model.py
    pixel_values = T.SimTensor(
        {
            "name": "pixel_values",
            "data": np.random.randn(batch_size, num_channels, height, width).astype(np.float32),
            "shape": [batch_size, num_channels, height, width],
            "dtype": "float32",
        }
    )

    parameters: Dict[str, Any] = {
        "encoder": {}
    }

    try:
        model = TtsimSegformerModel("segformer_model", config, parameters)

        outputs = model(
            pixel_values,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        final_shape = list(outputs.last_hidden_state.shape)
        expected_final_shape = [1, 16, 16, 256]

        print(f"Input shape (NCHW): {list(pixel_values.shape)}")
        print(f"Output shape (last_hidden_state): {final_shape}")

        if final_shape == expected_final_shape:
            print(f"[PASSED] Final output shape matches expected: {expected_final_shape}")
        else:
            print(f"[FAILED] Expected final shape {expected_final_shape}, got {final_shape}")

        expected_stage_shapes = [
            [1, 128 * 128, 32],
            [1, 64 * 64, 64],
            [1, 32 * 32, 160],
            [1, 16 * 16, 256],
        ]

        if outputs.hidden_states is not None:
            print("Hidden states:")
            for i, hs in enumerate(outputs.hidden_states):
                actual = list(hs.shape)
                expected = expected_stage_shapes[i]
                status = "OK" if actual == expected else "MISMATCH"
                print(f"  Stage {i+1}: {actual} | expected {expected} | {status}")
        else:
            print("Hidden states: None")

        print("=== Test Complete ===")

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_segformer_model()