# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from typing import Any, Tuple

# Force the root directory into the path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_attention import TtsimSegformerAttention
from workloads.segformer.tt.segformer_mix_ffn import TtsimSegformerMixFFN


class TtsimSegformerLayer:
    """This corresponds to the Transformer Block (Attention + MixFFN) for Polaris."""
    
    def __init__(self, name: str, config: Any, hidden_size: int, num_attention_heads: int, 
                 sequence_reduction_ratio: int, parameters: Any, mlp_ratio: int):
        self.name = name
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.config = config

        # 1. Pre-Attention LayerNorm (SAFE BROADCAST SHAPES)
        self.ln_1_w = T.SimTensor({"name": f"{name}_ln_1_w", "data": parameters["layer_norm_1"]["weight"].reshape(1, 1, hidden_size), "shape": [1, 1, hidden_size], "dtype": "float32"})
        self.ln_1_b = T.SimTensor({"name": f"{name}_ln_1_b", "data": parameters["layer_norm_1"]["bias"].reshape(1, 1, hidden_size), "shape": [1, 1, hidden_size], "dtype": "float32"})

        # 2. Attention Mechanism (no config argument)
        self.attention = TtsimSegformerAttention(
            name=f"{self.name}_attention",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters["attention"],
            sequence_reduction_ratio=sequence_reduction_ratio,
        )

        # 3. Pre-MLP LayerNorm (SAFE BROADCAST SHAPES)
        self.ln_2_w = T.SimTensor({"name": f"{name}_ln_2_w", "data": parameters["layer_norm_2"]["weight"].reshape(1, 1, hidden_size), "shape": [1, 1, hidden_size], "dtype": "float32"})
        self.ln_2_b = T.SimTensor({"name": f"{name}_ln_2_b", "data": parameters["layer_norm_2"]["bias"].reshape(1, 1, hidden_size), "shape": [1, 1, hidden_size], "dtype": "float32"})

        # 4. MixFFN (MLP) - needs config, in_features, hidden_features, out_features, parameters
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TtsimSegformerMixFFN(
            name=f"{self.name}_mlp",
            config=config,
            in_features=hidden_size,
            hidden_features=mlp_hidden_size,
            out_features=hidden_size,
            parameters=parameters["mlp"]
        )

    def __call__(self, hidden_states: Any, height: int, width: int, output_attentions: bool = False) -> Tuple[Any, ...]:
        # --- PHASE 1: ATTENTION ---
        # Safe Functional LayerNorm 1
        ln_1_out = F.LayerNorm(f"{self.name}_ln1", self.hidden_size)(hidden_states)
        ln_1_out = F.Mul(f"{self.name}_ln1_mul")(ln_1_out, self.ln_1_w)
        ln_1_out = F.Add(f"{self.name}_ln1_add")(ln_1_out, self.ln_1_b)

        # Run Attention
        self_attention_outputs = self.attention(
            ln_1_out,
            height,
            width,
            output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # First Residual Connection
        hidden_states = F.Add(f"{self.name}_res_1")(attention_output, hidden_states)

        # --- PHASE 2: MIX-FFN (MLP) ---
        # Safe Functional LayerNorm 2
        ln_2_out = F.LayerNorm(f"{self.name}_ln2", self.hidden_size)(hidden_states)
        ln_2_out = F.Mul(f"{self.name}_ln2_mul")(ln_2_out, self.ln_2_w)
        ln_2_out = F.Add(f"{self.name}_ln2_add")(ln_2_out, self.ln_2_b)

        # Run MixFFN
        mlp_output = self.mlp(ln_2_out, height, width)

        # Second Residual Connection
        layer_output = F.Add(f"{self.name}_res_2")(mlp_output, hidden_states)

        return (layer_output,) + outputs