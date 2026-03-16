# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_encoder import TtsimSegformerEncoder, TtsimBaseModelOutput


class TtsimSegformerModel:
    def __init__(self, name: str, config: Any, parameters: Any) -> None:
        self.name = name
        self.config = config

        # Initialize the hierarchical encoder
        self.encoder = TtsimSegformerEncoder(
            name=f"{self.name}_encoder",
            config=config,
            parameters=parameters["encoder"]
        )

    def __call__(
        self,
        pixel_values: Any,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Any, ...], TtsimBaseModelOutput]:

        # 1. Configuration setup
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 2. Input Preparation
        # In Polaris, we pass the NCHW [Batch, 3, Height, Width] tensor directly to the encoder.
        # The Patch Embeddings (TtsimConv) handle the initial NCHW -> NHWC conversion internally.

        # 3. Execute Encoder
        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 4. Final Output Formatting
        if return_dict:
            # encoder_outputs is TtsimBaseModelOutput
            assert isinstance(encoder_outputs, TtsimBaseModelOutput)
            sequence_output = encoder_outputs.last_hidden_state
            return TtsimBaseModelOutput(
                last_hidden_state=sequence_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        else:
            # encoder_outputs is Tuple
            assert isinstance(encoder_outputs, tuple)
            sequence_output = encoder_outputs[0]
            return (sequence_output,) + encoder_outputs[1:]