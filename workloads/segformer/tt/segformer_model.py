# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
from typing import Any, Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_encoder import TtsimSegformerEncoder, TtsimBaseModelOutput

class TtsimSegformerModel:
    def __init__(self, name: str, config: Any, parameters: Any) -> None:
        self.name = name
        self.config = config

        self.encoder = TtsimSegformerEncoder(
            name=f"{self.name}_encoder",
            config=config,
            parameters=parameters["encoder"],
        )

        # pre-create transpose + concat op
        self.pad_concat = F.ConcatX(f"{self.name}_pad_concat", axis=1)
        self.to_nhwc = F.Transpose(f"{self.name}_to_nhwc", perm=[0, 2, 3, 1])

    def _make_const_tensor(self, name: str, data: np.ndarray, dtype: str = "float32") -> T.SimTensor:
        return T.SimTensor(
            {
                "name": name,
                "data": data.astype(np.float32),
                "shape": list(data.shape),
                "dtype": dtype,
            }
        )

    def __call__(
        self,
        pixel_values: Any,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        min_channels: int = 8,
    ) -> Union[Tuple[Any, ...], TtsimBaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        shape = list(pixel_values.shape)
        if len(shape) != 4:
            raise ValueError(f"{self.name}: expected NCHW input [B, C, H, W], got {shape}")

        N, C, H, W = shape
        x = pixel_values

        # match TT-Metal model.py: pad NCHW channels to min 8 before NHWC permute
        if C < min_channels:
            pad_c = min_channels - C
            pad_tensor = self._make_const_tensor(
                f"{self.name}_pad_zeros",
                np.zeros((N, pad_c, H, W), dtype=np.float32),
            )
            x = self.pad_concat(x, pad_tensor)

        # NCHW -> NHWC
        x = self.to_nhwc(x)

        encoder_outputs = self.encoder(
            x,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            sequence_output = encoder_outputs[0]
            return (sequence_output,) + encoder_outputs[1:]

        return TtsimBaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )