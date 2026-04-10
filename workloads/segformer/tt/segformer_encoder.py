# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_layer import TtsimSegformerLayer
from workloads.segformer.tt.segformer_overlap_patch_embeddings import TtsimSegformerOverlapPatchEmbeddings

@dataclass
class TtsimBaseModelOutput:
    last_hidden_state: Any = None
    hidden_states: Any = None
    attentions: Any = None

    def __getitem__(self, idx):
        if idx == 0:
            return self.last_hidden_state
        elif idx == 1:
            return self.hidden_states
        elif idx == 2:
            return self.attentions
        else:
            raise IndexError("Index out of range")

class TtsimLayerNormSafe:
    """
    TT-Metal encoder applies stage layernorm on 3D [B, S, C] tensors.
    So LN params should broadcast as [1, 1, C].
    """
    def __init__(self, name: str, hidden_size: int, weight: Any, bias: Any) -> None:
        self.name = name
        self.hidden_size = hidden_size
        self.ln_w = T.SimTensor(
            {
                "name": f"{name}_w",
                "data": weight.reshape(1, 1, hidden_size).astype(np.float32),
                "shape": [1, 1, hidden_size],
                "dtype": "float32",
            }
        )
        self.ln_b = T.SimTensor(
            {
                "name": f"{name}_b",
                "data": bias.reshape(1, 1, hidden_size).astype(np.float32),
                "shape": [1, 1, hidden_size],
                "dtype": "float32",
            }
        )

    def __call__(self, x: Any) -> Any:
        out = F.LayerNorm(f"{self.name}_op", self.hidden_size)(x)
        out = F.Mul(f"{self.name}_mul")(out, self.ln_w)
        out = F.Add(f"{self.name}_add")(out, self.ln_b)
        return out

class TtsimSegformerEncoder:
    def __init__(self, name: str, config: Any, parameters: Any) -> None:
        self.name = name
        self.config = config

        self.patch_embeddings: List[Any] = []
        for i in range(config.num_encoder_blocks):
            self.patch_embeddings.append(
                TtsimSegformerOverlapPatchEmbeddings(
                    name=f"{self.name}_patch_embeddings_{i}",
                    parameters=parameters["patch_embeddings"][i],
                    stride=config.strides[i],
                    patch_size=config.patch_sizes[i],
                )
            )

        self.block: List[List[Any]] = []
        for i in range(config.num_encoder_blocks):
            layers: List[Any] = []
            for j in range(config.depths[i]):
                layers.append(
                    TtsimSegformerLayer(
                        name=f"{self.name}_block_{i}_layer_{j}",
                        config=config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        parameters=parameters["block"][i][j],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            self.block.append(layers)

        self.layer_norm: List[Any] = []
        for i in range(config.num_encoder_blocks):
            self.layer_norm.append(
                TtsimLayerNormSafe(
                    name=f"{self.name}_layer_norm_{i}",
                    hidden_size=config.hidden_sizes[i],
                    weight=parameters["layer_norm"][i]["weight"],
                    bias=parameters["layer_norm"][i]["bias"],
                )
            )

    def _get_shape_tensor(self, shape_list: List[int], name: str) -> T.SimTensor:
        return T.SimTensor(
            {
                "name": name,
                "data": np.array([int(s) for s in shape_list], dtype=np.int64),
                "shape": [len(shape_list)],
                "dtype": np.int64,
            }
        )

    def __call__(
        self,
        pixel_values: Any,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[Any, ...], TtsimBaseModelOutput]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = int(pixel_values.shape[0])
        hidden_states = pixel_values

        for idx in range(self.config.num_encoder_blocks):
            embedding_layer = self.patch_embeddings[idx]
            block_layers = self.block[idx]
            norm_layer = self.layer_norm[idx]

            # TT-Metal behavior:
            # input to patch embedding is 4D NHWC
            # output is 3D [B, S, C], plus H, W
            hidden_states, height, width = embedding_layer(hidden_states)

            # blocks operate on 3D [B, S, C] inside encoder
            for blk in block_layers:
                layer_outputs = blk(
                    hidden_states,
                    height,
                    width,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # stage layer norm still on 3D [B, S, C]
            hidden_states = norm_layer(hidden_states)

            # TT-Metal stores hidden_states before reshaping for next stage
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # between stages, TT-Metal reshapes to NHWC, not NCHW
            if idx != len(self.patch_embeddings) - 1:
                channels = int(self.config.hidden_sizes[idx])
                rs_shape = self._get_shape_tensor(
                    [batch_size, height, width, channels],
                    f"{self.name}_reshape_{idx}_shape",
                )
                hidden_states = F.Reshape(f"{self.name}_reshape_{idx}")(hidden_states, rs_shape)

            elif idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage:
                channels = int(self.config.hidden_sizes[idx])
                rs_shape = self._get_shape_tensor(
                    [batch_size, height, width, channels],
                    f"{self.name}_reshape_last_shape",
                )
                hidden_states = F.Reshape(f"{self.name}_reshape_last")(hidden_states, rs_shape)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return TtsimBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )