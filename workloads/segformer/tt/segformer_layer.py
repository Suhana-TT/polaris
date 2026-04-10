# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Any, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_attention import TtsimSegformerAttention
from workloads.segformer.tt.segformer_mix_ffn import TtsimSegformerMixFFN

class TtsimSegformerLayer:
    """Polaris SegFormer layer: LN -> Attention -> residual -> LN -> MixFFN -> residual."""

    def __init__(
        self,
        name: str,
        config: Any,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        parameters: Any,
        mlp_ratio: int,
    ):
        self.name = name
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.config = config

        
        self.ln_1_w = T.SimTensor(
            {
                "name": f"{name}_ln_1_w",
                "data": parameters["layer_norm_1"]["weight"].reshape(1, 1, 1, hidden_size),
                "shape": [1, 1, 1, hidden_size],
                "dtype": "float32",
            }
        )
        self.ln_1_b = T.SimTensor(
            {
                "name": f"{name}_ln_1_b",
                "data": parameters["layer_norm_1"]["bias"].reshape(1, 1, 1, hidden_size),
                "shape": [1, 1, 1, hidden_size],
                "dtype": "float32",
            }
        )

        self.attention = TtsimSegformerAttention(
            name=f"{self.name}_attention",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            parameters=parameters["attention"],
            sequence_reduction_ratio=sequence_reduction_ratio,
        )

        self.ln_2_w = T.SimTensor(
            {
                "name": f"{name}_ln_2_w",
                "data": parameters["layer_norm_2"]["weight"].reshape(1, 1, 1, hidden_size),
                "shape": [1, 1, 1, hidden_size],
                "dtype": "float32",
            }
        )
        self.ln_2_b = T.SimTensor(
            {
                "name": f"{name}_ln_2_b",
                "data": parameters["layer_norm_2"]["bias"].reshape(1, 1, 1, hidden_size),
                "shape": [1, 1, 1, hidden_size],
                "dtype": "float32",
            }
        )

        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TtsimSegformerMixFFN(
            name=f"{self.name}_mlp",
            config=config,
            in_features=hidden_size,
            hidden_features=mlp_hidden_size,
            out_features=hidden_size,
            parameters=parameters["mlp"],
        )

    def __call__(self, hidden_states: Any, height: int, width: int, output_attentions: bool = False) -> Tuple[Any, ...]:
        # hidden_states expected as [B, 1, S, C]

        # Pre-attention LayerNorm
        ln_1_out = F.LayerNorm(f"{self.name}_ln1", self.hidden_size)(hidden_states)
        ln_1_out = F.Mul(f"{self.name}_ln1_mul")(ln_1_out, self.ln_1_w)
        ln_1_out = F.Add(f"{self.name}_ln1_add")(ln_1_out, self.ln_1_b)

        # Attention
        self_attention_outputs = self.attention(
            ln_1_out,
            height,
            width,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # First residual
        hidden_states = F.Add(f"{self.name}_res_1")(attention_output, hidden_states)

        # Pre-MLP LayerNorm
        ln_2_out = F.LayerNorm(f"{self.name}_ln2", self.hidden_size)(hidden_states)
        ln_2_out = F.Mul(f"{self.name}_ln2_mul")(ln_2_out, self.ln_2_w)
        ln_2_out = F.Add(f"{self.name}_ln2_add")(ln_2_out, self.ln_2_b)

        # MixFFN
        mlp_output = self.mlp(ln_2_out, height, width)

        # Second residual
        layer_output = F.Add(f"{self.name}_res_2")(mlp_output, hidden_states)

        return (layer_output,) + outputs