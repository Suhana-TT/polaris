# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

# Standard repo path setup
current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from workloads.segformer.tt.segformer_model import TtsimSegformerModel
from workloads.segformer.tt.segformer_decode_head import TtsimSegformerDecodeHead
from workloads.segformer.tt.segformer_encoder import TtsimBaseModelOutput


@dataclass
class TtsimSemanticSegmenterOutput:
    loss: Any = None
    logits: Any = None
    hidden_states: Any = None
    attentions: Any = None


class TtsimSegformerForSemanticSegmentation:
    def __init__(self, name: str, config: Any, parameters: Any) -> None:
        self.name = name
        self.config = config

        # 1. Hierarchical Transformer Encoder (Backbone)
        self.segformer = TtsimSegformerModel(
            name=f"{self.name}_segformer",
            config=config,
            parameters=parameters["segformer"]
        )

        # 2. All-MLP Decoder Head
        self.decode_head = TtsimSegformerDecodeHead(
            name=f"{self.name}_decode_head",
            config=config,
            parameters=parameters["decode_head"]
        )

    def __call__(
        self,
        pixel_values: Any,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[Any, ...], TtsimSemanticSegmenterOutput]:

        # Standard configuration resolution
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        user_requested_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # --- PHASE 1: BACKBONE ---
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # Extract the hidden states tuple (Stages 1, 2, 3, 4)
        if return_dict and hasattr(outputs, 'hidden_states'):
            encoder_hidden_states = outputs.hidden_states
        elif isinstance(outputs, tuple) and len(outputs) > 1:
            encoder_hidden_states = outputs[1]
        else:
            # Fallback for mock objects
            encoder_hidden_states = getattr(outputs, 'hidden_states', outputs[1] if isinstance(outputs, tuple) else None)

        # --- PHASE 2: DECODER ---
        logits = self.decode_head(encoder_hidden_states)

        # Loss is skipped during inference
        loss = None

        if not return_dict:
            res: Tuple[Any, ...] = (logits,)
            if user_requested_hidden_states:
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    res = res + (outputs[1],)
                elif hasattr(outputs, 'hidden_states'):
                    res = res + (outputs.hidden_states,)
            if output_attentions:
                if isinstance(outputs, tuple) and len(outputs) > 2:
                    res = res + (outputs[2],)
                elif hasattr(outputs, 'attentions'):
                    res = res + (outputs.attentions,)
            return res

        return TtsimSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, 'hidden_states', None) if user_requested_hidden_states else None,
            attentions=getattr(outputs, 'attentions', None),
        )
