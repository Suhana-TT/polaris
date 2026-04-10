# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_common import TtsimConv

class TtsimSegformerDWConv:
    def __init__(self, name: str, parameters: dict, dim: int, activation=None):
        self.name = name
        self.dim = int(dim)

        conv_params = parameters["dwconv"] if "dwconv" in parameters else parameters

        self.dwconv = TtsimConv(
            name=f"{self.name}_dwconv",
            strides=[1, 1, 1, 1],
            parameters=conv_params,
            kernel_size=[3, 3],
            groups=self.dim,
        )

        self.to_nchw_t = F.Transpose(f"{self.name}_to_nchw_t", perm=[0, 2, 1])
        self.to_seq_t = F.Transpose(f"{self.name}_to_seq_t", perm=[0, 2, 1])

        self.gelu = None if activation is None else F.Gelu(f"{self.name}_gelu")

    def _get_shape_tensor(self, shape_list, name):
        return T.SimTensor(
            {
                "name": name,
                "data": np.array([int(s) for s in shape_list], dtype=np.int64),
                "shape": [len(shape_list)],
                "dtype": np.int64,
            }
        )

    def __call__(self, hidden_states, height: int, width: int):
        x = hidden_states[0] if isinstance(hidden_states, (list, tuple)) else hidden_states
        shape = list(x.shape)

        if len(shape) == 3:
            batch_size, seq_len, num_channels = shape
        elif len(shape) == 4:
            batch_size, one, seq_len, num_channels = shape
            if one != 1:
                raise ValueError(f"{self.name}: expected [B, 1, S, C], got {shape}")
            x = F.Reshape(f"{self.name}_squeeze_seq")(
                x,
                self._get_shape_tensor([batch_size, seq_len, num_channels], f"{self.name}_shape_3d"),
            )
        else:
            raise ValueError(f"{self.name}: expected 3D or 4D input, got {shape}")

        if int(seq_len) != int(height * width):
            raise ValueError(
                f"{self.name}: seq_len {seq_len} does not match height*width {height * width}"
            )

        # [B, S, C] -> [B, C, S] -> [B, C, H, W]
        x = self.to_nchw_t(x)
        x = F.Reshape(f"{self.name}_to_nchw_rs")(
            x,
            self._get_shape_tensor([batch_size, num_channels, height, width], f"{self.name}_shape_nchw"),
        )

        x = self.dwconv(x)

        if self.gelu is not None:
            x = self.gelu(x)

        # [B, C, H, W] -> [B, C, S] -> [B, S, C]
        x = F.Reshape(f"{self.name}_to_seq_rs")(
            x,
            self._get_shape_tensor([batch_size, num_channels, height * width], f"{self.name}_shape_chw_seq"),
        )
        x = self.to_seq_t(x)

        return x