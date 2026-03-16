# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F


class TtsimConv:
    def __init__(self, name: str, strides: List[int], parameters: Dict[str, Any], 
                 kernel_size: Optional[List[int]] = None, groups: int = 1):
        self.name = name
        self.strides = strides
        self.dtype = "float32"
        self.bias: Optional[T.SimTensor] = None  # Initialize with type hint

        raw_w = parameters["weight"]

        # Ensure 4D shape [Out, In, H, W]
        if len(raw_w.shape) == 3:
            raw_w = raw_w.reshape(raw_w.shape[0], 1, raw_w.shape[1], raw_w.shape[2])

        out_c = int(raw_w.shape[0])
        weight_in_c = int(raw_w.shape[1])  # This is (in_channels / groups)

        # --- THE FIX: Calculate the REAL total in_channels ---
        actual_in_c = weight_in_c * int(groups)

        def flatten(lst: Any) -> List[int]:
            flat: List[int] = []
            for item in lst:
                if isinstance(item, (list, tuple)):
                    flat.extend(flatten(item))
                else:
                    flat.append(int(item))
            return flat

        if kernel_size is not None:
            k_dims = flatten(kernel_size)
            k_h, k_w = k_dims[0], k_dims[1]
        else:
            k_h = int(raw_w.shape[2]) if len(raw_w.shape) > 2 else 1
            k_w = int(raw_w.shape[3]) if len(raw_w.shape) > 3 else 1

        flat_shape = [out_c, weight_in_c, k_h, k_w]

        self.weight = T.SimTensor({
            "name": f"{name}_w",
            "data": raw_w,
            "shape": flat_shape,
            "dtype": self.dtype
        })

        pad_val = int(self.strides[2]) if len(self.strides) > 2 else 1

        self.conv_op = F.Conv2d(
            f"{name}_op",
            in_channels=actual_in_c,  # Pass the true channel count!
            out_channels=out_c,
            kernel_size=k_h,
            stride=int(self.strides[0]),
            padding=pad_val,
            groups=int(groups)
        )

        self.conv_op.weight = self.weight

        if "bias" in parameters:
            raw_b = parameters["bias"]
            b_shape = [1, out_c, 1, 1]
            self.bias = T.SimTensor({
                "name": f"{name}_b",
                "data": raw_b.reshape(b_shape),
                "shape": b_shape,
                "dtype": self.dtype
            })

    def __call__(self, x: Any) -> Any:
        out = self.conv_op(x)
        if self.bias is not None:
            out = F.Add(f"{self.name}_add")(out, self.bias)
        return out