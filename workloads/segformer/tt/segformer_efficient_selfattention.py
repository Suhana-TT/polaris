# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
from typing import Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
from workloads.segformer.tt.segformer_common import TtsimConv


class TtsimSegformerEfficientSelfAttention:
    def __init__(self, name: str, hidden_size: int, num_attention_heads: int, parameters: Any, sequence_reduction_ratio: int):
        self.name = name
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.sr_ratio = sequence_reduction_ratio
        self.dtype = "float32"

        def _get_lin(n: str, p: dict) -> Tuple[T.SimTensor, T.SimTensor]:
            w_data = p["weight"].T if p["weight"].shape[0] == self.hidden_size else p["weight"]
            w = T.SimTensor({"name": f"{n}_w", "data": w_data, "shape": list(w_data.shape), "dtype": self.dtype})
            b = T.SimTensor({"name": f"{n}_b", "data": p["bias"].reshape(1, -1), "shape": [1, p["bias"].shape[0]], "dtype": self.dtype})
            return w, b

        self.q_w, self.q_b = _get_lin(f"{name}_q", parameters["query"])
        self.k_w, self.k_b = _get_lin(f"{name}_k", parameters["key"])
        self.v_w, self.v_b = _get_lin(f"{name}_v", parameters["value"])

        self.scale_tensor = T.SimTensor({
            "name": f"{name}_scale",
            "data": np.array([self.head_size**-0.5], dtype=np.float32),
            "shape": [1],
            "dtype": self.dtype
        })

        if self.sr_ratio > 1:
            self.sr = TtsimConv(f"{name}_sr", [self.sr_ratio, self.sr_ratio, 0, 0], parameters["sr"])
            self.ln_w = T.SimTensor({"name": f"{name}_ln_w", "data": parameters["layer_norm"]["weight"], "shape": list(parameters["layer_norm"]["weight"].shape), "dtype": self.dtype})
            self.ln_b = T.SimTensor({"name": f"{name}_ln_b", "data": parameters["layer_norm"]["bias"], "shape": list(parameters["layer_norm"]["bias"].shape), "dtype": self.dtype})

    def _linear(self, x: Any, w: T.SimTensor, b: T.SimTensor, name: str) -> Any:
        mm = F.MatMul(f"{name}_mm")(x, w)
        return F.Add(f"{name}_add")(mm, b)

    def _get_shape_tensor(self, shape_list: list, name: str) -> T.SimTensor:
        return T.SimTensor({
            "name": name,
            "data": np.array(shape_list, dtype=np.int64),
            "shape": [len(shape_list)],
            "dtype": np.int64
        })

    def __call__(self, hidden_states: Any, height: int, width: int) -> Tuple[Any]:
        x = hidden_states[0] if isinstance(hidden_states, list) else hidden_states
        B, S, C = x.shape

        # 1. Query
        q = self._linear(x, self.q_w, self.q_b, f"{self.name}_q")
        q_shape = self._get_shape_tensor([B, S, self.num_attention_heads, self.head_size], f"{self.name}_q_shape")
        q = F.Reshape(f"{self.name}_q_rs")(q, q_shape)
        q = F.Transpose(f"{self.name}_q_pm", perm=[0, 2, 1, 3])(q)

        # 2. Key/Value with SR
        kv = x
        if self.sr_ratio > 1:
            kv_shape1 = self._get_shape_tensor([B, height, width, C], f"{self.name}_kv_shape1")
            kv = F.Reshape(f"{self.name}_kv_rs1")(kv, kv_shape1)
            kv = F.Transpose(f"{self.name}_kv_pm1", perm=[0, 3, 1, 2])(kv)
            kv = self.sr(kv)
            kv = F.Transpose(f"{self.name}_kv_pm2", perm=[0, 2, 3, 1])(kv)
            kv_shape2 = self._get_shape_tensor([B, -1, C], f"{self.name}_kv_shape2")
            kv = F.Reshape(f"{self.name}_kv_rs2")(kv, kv_shape2)
            # FIX: Separate normalization and parameter application
            kv = F.LayerNorm(f"{self.name}_ln", C)(kv)
            kv = F.Mul(f"{self.name}_ln_mul")(kv, self.ln_w)
            kv = F.Add(f"{self.name}_ln_add")(kv, self.ln_b)

        k = self._linear(kv, self.k_w, self.k_b, f"{self.name}_k")
        k_shape = self._get_shape_tensor([B, -1, self.num_attention_heads, self.head_size], f"{self.name}_k_shape")
        k = F.Reshape(f"{self.name}_k_rs")(k, k_shape)
        k = F.Transpose(f"{self.name}_k_pm1", perm=[0, 2, 3, 1])(k)

        v = self._linear(kv, self.v_w, self.v_b, f"{self.name}_v")
        v_shape = self._get_shape_tensor([B, -1, self.num_attention_heads, self.head_size], f"{self.name}_v_shape")
        v = F.Reshape(f"{self.name}_v_rs")(v, v_shape)
        v = F.Transpose(f"{self.name}_v_pm", perm=[0, 2, 1, 3])(v)

        # 3. Attention
        attn = F.MatMul(f"{self.name}_mm1")(q, k)
        attn = F.Mul(f"{self.name}_sc")(attn, self.scale_tensor)
        attn = F.Softmax(f"{self.name}_sm", dim=-1)(attn)
        out = F.MatMul(f"{self.name}_mm2")(attn, v)
        out = F.Transpose(f"{self.name}_o_pm", perm=[0, 2, 1, 3])(out)
        out_shape = self._get_shape_tensor([B, S, C], f"{self.name}_out_shape")
        out = F.Reshape(f"{self.name}_o_rs")(out, out_shape)

        return (out,)