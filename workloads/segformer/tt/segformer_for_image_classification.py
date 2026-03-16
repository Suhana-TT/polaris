# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np

# Standard path logic
current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_model import TtsimSegformerModel


@dataclass
class TtsimSegFormerImageClassifierOutput:
    loss: Any = None
    logits: Any = None
    hidden_states: Any = None
    attentions: Any = None


class TtsimSegformerForImageClassification:
    def __init__(self, name: str, config: Any, parameters: Any):
        self.name = name
        self.config = config
        self.num_labels = config.num_labels

        self.segformer = TtsimSegformerModel(
            name=f"{self.name}_segformer",
            config=config,
            parameters=parameters["segformer"]
        )

        self.classifier_w = T.SimTensor({
            "name": f"{self.name}_classifier_w",
            "data": parameters["classifier"]["weight"],
            "shape": list(parameters["classifier"]["weight"].shape),
            "dtype": "float32"
        })

        b_shape = [1, int(parameters["classifier"]["bias"].shape[0])]
        self.classifier_b = T.SimTensor({
            "name": f"{self.name}_classifier_b",
            "data": parameters["classifier"]["bias"].reshape(b_shape),
            "shape": b_shape,
            "dtype": "float32"
        })

    def _get_shape_tensor(self, shape_list: list, name: str) -> T.SimTensor:
        return T.SimTensor({
            "name": name,
            "data": np.array([int(s) for s in shape_list], dtype=np.int64),
            "shape": [len(shape_list)],
            "dtype": np.int64
        })

    def __call__(
        self,
        pixel_values: Any,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[Any, ...], TtsimSegFormerImageClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.segformer(pixel_values)
        sequence_output = outputs[0]  # type: ignore

        batch_size = int(sequence_output.shape[0])
        hidden_size = int(self.config.hidden_sizes[-1])

        # 1. Prepare sequence [B, Seq_Len, Hidden]
        if len(sequence_output.shape) == 4:
            sequence_output = F.Transpose(f"{self.name}_gap_tr", perm=[0, 2, 3, 1])(sequence_output)
            seq_len = int(sequence_output.shape[1] * sequence_output.shape[2])
            rs_shape = self._get_shape_tensor([batch_size, seq_len, hidden_size], f"{self.name}_gap_rs")
            sequence_output = F.Reshape(f"{self.name}_gap_reshape")(sequence_output, rs_shape)
        else:
            seq_len = int(sequence_output.shape[1])

        # 2. Transpose for Pooling [B, Hidden, Seq_Len]
        sequence_output = F.Transpose(f"{self.name}_pool_tr", perm=[0, 2, 1])(sequence_output)

        # 3. GEMM-based Pooling: [B, Hidden, Seq_Len] @ [Seq_Len, 1] -> [B, Hidden, 1]
        pool_weights = T.SimTensor({
            "name": f"{self.name}_pool_weights",
            "data": np.ones((seq_len, 1), dtype=np.float32) * (1.0 / float(seq_len)),
            "shape": [seq_len, 1],
            "dtype": "float32"
        })
        pooled_output = F.MatMul(f"{self.name}_gap_matmul")(sequence_output, pool_weights)

        # 4. Reshape to [B, Hidden]
        final_pool_shape = self._get_shape_tensor([batch_size, hidden_size], f"{self.name}_pool_rs2")
        pooled_output = F.Reshape(f"{self.name}_pool_reshape2")(pooled_output, final_pool_shape)

        # 5. Classification
        logits = F.MatMul(f"{self.name}_classifier_matmul")(pooled_output, self.classifier_w)
        logits = F.Add(f"{self.name}_classifier_add")(logits, self.classifier_b)

        return TtsimSegFormerImageClassifierOutput(logits=logits)