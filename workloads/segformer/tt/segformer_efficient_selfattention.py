# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
from typing import Any, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from workloads.segformer.tt.segformer_common import TtsimConv

class TtsimSegformerEfficientSelfAttention(SimNN.Module):
    def __init__(
        self,
        name: str,
        hidden_size: int,
        num_attention_heads: int,
        parameters: Any,
        sequence_reduction_ratio: int,
    ):
        super().__init__()
        self.name = name
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.sr_ratio = sequence_reduction_ratio
        self.dtype = "float32"

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention heads "
                f"({self.num_attention_heads})"
            )

        self.q_w, self.q_b = self._make_linear_params(f"{name}.query", parameters["query"])
        self.k_w, self.k_b = self._make_linear_params(f"{name}.key", parameters["key"])
        self.v_w, self.v_b = self._make_linear_params(f"{name}.value", parameters["value"])

        self.scale_tensor = self._const_tensor(
            f"{name}.scale",
            np.array([self.head_size**-0.5], dtype=np.float32),
        )

        if self.sr_ratio > 1:
            self.sr = TtsimConv(f"{name}.sr", [self.sr_ratio, self.sr_ratio, 0, 0], parameters["sr"])
            self.ln_w = self._const_tensor(f"{name}.layer_norm.weight", parameters["layer_norm"]["weight"])
            self.ln_b = self._const_tensor(f"{name}.layer_norm.bias", parameters["layer_norm"]["bias"])

        # pre-registered ops
        self.q_mm = F.MatMul(f"{name}.q_mm")
        self.q_add = F.Add(f"{name}.q_add")
        self.q_rs = F.Reshape(f"{name}.q_rs")
        self.q_pm = F.Transpose(f"{name}.q_pm", perm=[0, 2, 1, 3])

        self.k_mm = F.MatMul(f"{name}.k_mm")
        self.k_add = F.Add(f"{name}.k_add")
        self.k_rs = F.Reshape(f"{name}.k_rs")
        self.k_pm = F.Transpose(f"{name}.k_pm", perm=[0, 2, 3, 1])

        self.v_mm = F.MatMul(f"{name}.v_mm")
        self.v_add = F.Add(f"{name}.v_add")
        self.v_rs = F.Reshape(f"{name}.v_rs")
        self.v_pm = F.Transpose(f"{name}.v_pm", perm=[0, 2, 1, 3])

        self.in_rs = F.Reshape(f"{name}.in_rs")
        self.kv_rs1 = F.Reshape(f"{name}.kv_rs1")
        self.kv_pm1 = F.Transpose(f"{name}.kv_pm1", perm=[0, 3, 1, 2])
        self.kv_pm2 = F.Transpose(f"{name}.kv_pm2", perm=[0, 2, 3, 1])
        self.kv_rs2 = F.Reshape(f"{name}.kv_rs2")

        self.ln = F.LayerNorm(f"{name}.ln", hidden_size)
        self.ln_mul = F.Mul(f"{name}.ln_mul")
        self.ln_add = F.Add(f"{name}.ln_add")

        self.attn_mm = F.MatMul(f"{name}.attn_mm")
        self.attn_scale = F.Mul(f"{name}.attn_scale")
        self.attn_sm = F.Softmax(f"{name}.attn_sm", dim=-1)
        self.ctx_mm = F.MatMul(f"{name}.ctx_mm")

        self.o_pm = F.Transpose(f"{name}.o_pm", perm=[0, 2, 1, 3])
        self.o_rs3d = F.Reshape(f"{name}.o_rs3d")
        self.o_rs4d = F.Reshape(f"{name}.o_rs4d")

        super().link_op2module()

    def _const_tensor(self, name: str, data: np.ndarray) -> T.SimTensor:
        t = T.SimTensor(
            {
                "name": name,
                "data": np.asarray(data, dtype=np.float32),
                "shape": list(np.asarray(data).shape),
                "dtype": self.dtype,
                "op_in": [],
            }
        )
        t.link_module = self
        self._tensors[t.name] = t
        return t

    def _shape_tensor(self, name: str, shape_list: list[int]) -> T.SimTensor:
        t = T.SimTensor(
            {
                "name": name,
                "data": np.array(shape_list, dtype=np.int64),
                "shape": [len(shape_list)],
                "dtype": np.int64,
                "op_in": [],
            }
        )
        t.link_module = self
        self._tensors[t.name] = t
        return t

    def _make_linear_params(self, prefix: str, p: dict) -> Tuple[T.SimTensor, T.SimTensor]:
        w_data = p["weight"].T if p["weight"].shape[0] == self.hidden_size else p["weight"]
        w = self._const_tensor(f"{prefix}.weight", w_data)
        b = self._const_tensor(f"{prefix}.bias", p["bias"].reshape(1, -1))
        return w, b

    def _linear(self, x: Any, w: T.SimTensor, b: T.SimTensor, mm_op: Any, add_op: Any) -> Any:
        return add_op(mm_op(x, w), b)

    def __call__(self, hidden_states: Any, height: int, width: int, output_attentions: bool = False) -> Tuple[Any]:
        x = hidden_states[0] if isinstance(hidden_states, (list, tuple)) else hidden_states
        shape = list(x.shape)

        if len(shape) != 4:
            raise ValueError(f"{self.name}: expected 4D input [B, 1, S, C], got {shape}")

        B, one, S, C = shape
        if one != 1:
            raise ValueError(f"{self.name}: expected second dim == 1, got {shape}")

        # [B,1,S,C] -> [B,S,C]
        x = self.in_rs(x, self._shape_tensor(f"{self.name}.in_shape3d", [B, S, C]))

        # Query: [B,S,C] -> [B,H,S,D]
        q = self._linear(x, self.q_w, self.q_b, self.q_mm, self.q_add)
        q = self.q_rs(q, self._shape_tensor(f"{self.name}.q_shape", [B, S, self.num_attention_heads, self.head_size]))
        q = self.q_pm(q)

        # KV path
        kv = x
        if self.sr_ratio > 1:
            kv = self.kv_rs1(kv, self._shape_tensor(f"{self.name}.kv_hw_shape", [B, height, width, C]))
            kv = self.kv_pm1(kv)  # [B,C,H,W]
            kv = self.sr(kv)
            kv = self.kv_pm2(kv)  # [B,H',W',C]
            kv = self.kv_rs2(kv, self._shape_tensor(f"{self.name}.kv_seq_shape", [B, -1, C]))
            kv = self.ln(kv)
            kv = self.ln_mul(kv, self.ln_w)
            kv = self.ln_add(kv, self.ln_b)

        # Key: [B,S',C] -> [B,H,D,S']
        k = self._linear(kv, self.k_w, self.k_b, self.k_mm, self.k_add)
        k = self.k_rs(k, self._shape_tensor(f"{self.name}.k_shape", [B, -1, self.num_attention_heads, self.head_size]))
        k = self.k_pm(k)

        # Value: [B,S',C] -> [B,H,S',D]
        v = self._linear(kv, self.v_w, self.v_b, self.v_mm, self.v_add)
        v = self.v_rs(v, self._shape_tensor(f"{self.name}.v_shape", [B, -1, self.num_attention_heads, self.head_size]))
        v = self.v_pm(v)

        # Attention
        attn = self.attn_mm(q, k)                 # [B,H,S,S']
        attn = self.attn_scale(attn, self.scale_tensor)
        attn = self.attn_sm(attn)

        out = self.ctx_mm(attn, v)               # [B,H,S,D]
        out = self.o_pm(out)                     # [B,S,H,D]
        out = self.o_rs3d(out, self._shape_tensor(f"{self.name}.out_shape3d", [B, S, C]))
        out = self.o_rs4d(out, self._shape_tensor(f"{self.name}.out_shape4d", [B, 1, S, C]))

        if output_attentions:
            return (out, attn)
        return (out,)