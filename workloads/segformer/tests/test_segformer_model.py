# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys
import numpy as np
from typing import Any, Dict, Optional

# Force pathing
sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")
import ttsim.front.functional.tensor_op as T
from workloads.segformer.common import load_config  # Real config loader
from workloads.segformer.tt.segformer_encoder import TtsimBaseModelOutput
import workloads.segformer.tt.segformer_model as class_module


# --- MOCK ENCODER (temporary until real encoder is fully integrated) ---
class MockEncoder:
    """Temporary mock encoder - returns fake output with correct shape"""
    def __init__(self, name: str, config: Any, parameters: Any) -> None:
        self.name = name
        self.config = config

    def __call__(self, pixel_values: Any, output_attentions: Optional[bool], 
                 output_hidden_states: Optional[bool], return_dict: Optional[bool]) -> TtsimBaseModelOutput:
        batch = int(pixel_values.shape[0])
        
        # Output in NHWC format: [N, H, W, C] to match tt-metal
        final_state = T.SimTensor({
            "name": f"{self.name}_output",
            "data": np.random.randn(batch, 16, 16, 256).astype(np.float32),
            "shape": [batch, 16, 16, 256],
            "dtype": "float32"
        })

        # Return hidden_states only if requested
        if output_hidden_states:
            hidden_states = (final_state, final_state, final_state, final_state)
        else:
            hidden_states = None

        return TtsimBaseModelOutput(
            last_hidden_state=final_state,
            hidden_states=hidden_states,
            attentions=None
        )


# Inject mock encoder using setattr
setattr(class_module, 'TtsimSegformerEncoder', MockEncoder)

# Import after patching
from workloads.segformer.tt.segformer_model import TtsimSegformerModel


def test_segformer_model() -> None:
    print("=== Starting Polaris Segformer Model Test ===\n")

    # --- 1. Load REAL config (like tt-metal) ---
    config = load_config("configs/segformer_semantic_config.json")

    # Print config values to verify
    print(f"[CONFIG] num_channels: {config.num_channels}")
    print(f"[CONFIG] num_encoder_blocks: {config.num_encoder_blocks}")
    print(f"[CONFIG] hidden_sizes: {config.hidden_sizes}")
    print(f"[CONFIG] depths: {config.depths}")
    print(f"[CONFIG] output_hidden_states: {config.output_hidden_states}")
    print(f"[CONFIG] use_return_dict: {config.use_return_dict}")
    print()

    # --- 2. Create input tensor ---
    batch_size = 1
    num_channels = config.num_channels  # Use from config
    height = 512
    width = 512

    pixel_values = T.SimTensor({
        "name": "pixel_values",
        "data": np.random.randn(batch_size, num_channels, height, width).astype(np.float32),
        "shape": [batch_size, num_channels, height, width],
        "dtype": "float32"
    })

    # --- 3. Create parameters (mock for now) ---
    parameters: Dict[str, Any] = {
        "encoder": {}
    }

    # --- 4. Create and run model ---
    model = TtsimSegformerModel("segformer_model", config, parameters)
    outputs = model(
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    )

    # --- 5. Print results (matching tt-metal format) ---
    print(f"Input shape (pixel_values): {pixel_values.shape}")
    print(f"Output shape (sequence_output): {outputs.last_hidden_state.shape}")  # type: ignore

    if outputs.hidden_states is not None:  # type: ignore
        print(f"Total hidden states returned: {len(outputs.hidden_states)}")  # type: ignore
    else:
        print(f"Total hidden states returned: 0 (hidden_states is None)")

    # --- 6. Verify output shape ---
    expected_shape = [1, 16, 16, 256]
    actual_shape = list(outputs.last_hidden_state.shape)  # type: ignore

    print()
    if actual_shape == expected_shape:
        print(f"[PASSED] Output shape matches expected: {expected_shape}")
    else:
        print(f"[FAILED] Expected {expected_shape}, got {actual_shape}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_segformer_model()