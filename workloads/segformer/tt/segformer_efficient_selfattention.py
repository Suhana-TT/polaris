# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_common import Conv


def create_qkv_heads(tensor, num_heads, memory_config):
    """
    Manual implementation of nlp_create_qkv_heads_segformer.
    """
    if len(tensor.shape) == 4:
        batch, _, seq_len, hidden_size = tensor.shape
    else:
        batch, seq_len, hidden_size = tensor.shape

    head_size = hidden_size // num_heads
    tensor = ttnn.reshape(tensor, (batch, seq_len, num_heads, head_size))
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))

    return tensor


def concat_heads(tensor, memory_config):
    """
    Manual implementation of nlp_concat_heads.
    """
    batch, num_heads, seq_len, head_size = tensor.shape
    hidden_size = num_heads * head_size
    tensor = ttnn.permute(tensor, (0, 2, 1, 3))
    tensor = ttnn.reshape(tensor, (batch, seq_len, hidden_size))

    return tensor


class TtSegformerEfficientSelfAttention:
    def __init__(self, hidden_size, num_attention_heads, parameters, sequence_reduction_ratio):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sr_ratio = sequence_reduction_ratio

        if sequence_reduction_ratio > 1:
            self.sr = Conv([sequence_reduction_ratio, sequence_reduction_ratio, 0, 0], parameters["sr"])
            self.layer_norm_weight = parameters["layer_norm"]["weight"]
            self.layer_norm_bias = parameters["layer_norm"]["bias"]

        self.query_weight = parameters["query"]["weight"]
        self.query_bias = parameters["query"]["bias"]
        self.key_weight = parameters["key"]["weight"]
        self.key_bias = parameters["key"]["bias"]
        self.value_weight = parameters["value"]["weight"]
        self.value_bias = parameters["value"]["bias"]

    def __call__(
        self,
        device,
        hidden_states: ttnn.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ):
        if len(hidden_states.shape) == 4:
            batch_size, __, seq_len, hidden_size = hidden_states.shape
        elif len(hidden_states.shape) == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape

        # Store the number of channels for later use
        num_channels = hidden_size

        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

        query = ttnn.linear(
            hidden_states,
            self.query_weight,
            bias=self.query_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        if self.num_attention_heads == 1:
            query_layer = query
        else:
            query_layer = create_qkv_heads(query, self.num_attention_heads, ttnn.L1_MEMORY_CONFIG)

        if self.sr_ratio > 1:
            # Get current shape info
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, num_channels = hidden_states.shape
            elif len(hidden_states.shape) == 4:
                batch_size, __, seq_len, num_channels = hidden_states.shape

            # First reshape to remove the extra dimension if present
            if len(hidden_states.shape) == 4:
                hidden_states = ttnn.reshape(hidden_states, (batch_size, seq_len, num_channels))

            # Reshape from [batch, seq_len, channels] to [batch, height, width, channels] (NHWC)
            hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, num_channels))

            # Convert NHWC to NCHW for conv2d: [batch, height, width, channels] -> [batch, channels, height, width]
            hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

            # Convert to ROW_MAJOR layout for conv2d
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

            # Apply sequence reduction convolution (expects NCHW)
            hidden_states, out_h, out_w = self.sr(device, hidden_states)
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Conv output is NCHW: [batch, channels, out_h, out_w]
            # Convert back to sequence format: first to NHWC, then flatten
            # NCHW -> NHWC: [batch, channels, out_h, out_w] -> [batch, out_h, out_w, channels]
            hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))

            # Reshape to sequence format: [batch, out_h, out_w, channels] -> [batch, 1, out_h*out_w, channels]
            new_seq_len = out_h * out_w
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, new_seq_len, num_channels))

            # Apply layer norm
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=self.layer_norm_weight,
                bias=self.layer_norm_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
            )

        key = ttnn.linear(
            hidden_states,
            self.key_weight,
            bias=self.key_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        if self.num_attention_heads == 1:
            key_layer = key
        else:
            key_layer = create_qkv_heads(key, self.num_attention_heads, ttnn.L1_MEMORY_CONFIG)

        key_layer = ttnn.permute(key_layer, (0, 1, 3, 2))

        value = ttnn.linear(
            hidden_states,
            self.value_weight,
            bias=self.value_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        ttnn.deallocate(hidden_states)

        if self.num_attention_heads == 1:
            value_layer = value
        else:
            value_layer = create_qkv_heads(value, self.num_attention_heads, ttnn.L1_MEMORY_CONFIG)

        query_layer = ttnn.to_layout(query_layer, ttnn.TILE_LAYOUT)

        attention_scores = ttnn.matmul(
            query_layer,
            key_layer,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        ttnn.deallocate(query_layer)
        ttnn.deallocate(key_layer)

        scale_value = self.attention_head_size ** -0.5
        attention_scores = ttnn.multiply(attention_scores, scale_value)

        attention_probs = ttnn.softmax(attention_scores, dim=-1, numeric_stable=False)

        ttnn.deallocate(attention_scores)

        attention_probs = ttnn.to_layout(attention_probs, ttnn.TILE_LAYOUT)

        context_layer = ttnn.matmul(
            attention_probs,
            value_layer,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        ttnn.deallocate(value)
        ttnn.deallocate(value_layer)

        if not output_attentions:
            ttnn.deallocate(attention_probs)
        else:
            attention_probs = ttnn.to_memory_config(attention_probs, ttnn.L1_MEMORY_CONFIG)

        if self.num_attention_heads > 1:
            context_layer = ttnn.to_memory_config(context_layer, ttnn.L1_MEMORY_CONFIG)
            context_layer = concat_heads(context_layer, ttnn.L1_MEMORY_CONFIG)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs