# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Optimized DINOv2 and SigLIP encoders for OpenVLA - Polaris Version.
"""

import numpy as np
import ttsim.front.ttnn as ttnn

from typing import Any, cast
import numpy.typing as npt

# ============================================================================
# TTNN Helper Function for Polaris
# ============================================================================

def ttnn_from_numpy(array: np.ndarray, dtype=None, layout=None, device=None):
    """
    Create TTNN tensor from numpy array.
    Polaris version - uses ttnn.Tensor() instead of ttnn.from_numpy().

    Order: Tensor(with device) → typecast → to_layout
    """
    if array.dtype not in [np.float32, np.float16]:
        array = array.astype(np.float32)

    array = np.ascontiguousarray(array)

    if device is not None:
        tensor = ttnn.Tensor(array, device=device)
    else:
        tensor = ttnn.Tensor(array)

    if dtype is not None:
        tensor = ttnn.typecast(tensor, dtype)

    if layout is not None:
        tensor = ttnn.to_layout(tensor, layout)

    return tensor


# ============================================================================
# Patch Embedding Functions
# ============================================================================

def vit_patch_embeddings_weight_vars(
    config,
    pixel_values,
    proj_weight,
    proj_bias,
    patch_size=16,
):
    """ViT patch embedding with fold operation."""
    batch_size, img_h, img_w, img_c = pixel_values.shape
    patch_count = img_h // patch_size
    patch_count_all = int(patch_count * patch_count)
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(
        pixel_values,
        (batch_size, img_h, img_w // patch_size, 4 * patch_size),
    )
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    patch_embedding_output = ttnn.linear(
        pixel_values,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(pixel_values)

    patch_embedding_output = ttnn.to_layout(
        patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    patch_embedding_output = ttnn.reshape(
        patch_embedding_output, (batch_size, patch_count_all, -1)
    )
    return patch_embedding_output


# ============================================================================
# SigLIP Vision Blocks
# ============================================================================

def siglip_patch_embeddings(pixel_values, *, parameters):
    """SigLIP patch embedding with 14x14 patches."""
    batch_size, img_h, img_w, img_c = pixel_values.shape
    patch_size = 14
    patch_count = img_h // patch_size
    patch_count_all = int(patch_count * patch_count)
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(
        pixel_values,
        (batch_size, img_h, img_w // patch_size, 4 * patch_size),
    )
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    patch_embedding_output = ttnn.linear(
        pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(pixel_values)

    patch_embedding_output = ttnn.to_layout(
        patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    patch_embedding_output = ttnn.reshape(
        patch_embedding_output, (batch_size, patch_count_all, -1)
    )
    return patch_embedding_output


def siglip_attention(hidden_states, attention_mask, parameters):
    """SigLIP multi-head attention."""
    num_heads = 16
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.reallocate(hidden_states)

    query, key, value = _split_qkv_and_reshape_heads(
        query_key_value, num_heads=num_heads
    )
    ttnn.deallocate(query_key_value)

    scale = 1.0 / (head_size ** 0.5)
    query = ttnn.multiply(query, scale)
    value = ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.softmax(attention_scores)
    ttnn.deallocate(attention_scores)

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = _concatenate_heads(context_layer, num_heads=num_heads)

    self_output = ttnn.linear(
        context_layer,
        parameters.proj.weight,
        bias=parameters.proj.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(context_layer)

    return self_output


def siglip_intermediate(hidden_states, *, parameters):
    """SigLIP MLP intermediate layer with GELU."""
    return ttnn.linear(
        hidden_states,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation="gelu",
    )


def siglip_output(hidden_states, residual, *, parameters):
    """SigLIP MLP output layer with residual."""
    output = ttnn.linear(
        hidden_states,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(hidden_states)
    output = ttnn.add(output, residual, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    return output


def siglip_feedforward(hidden_states, attention_output, *, parameters):
    """SigLIP feedforward (MLP) block."""
    intermediate = siglip_intermediate(hidden_states, parameters=parameters.mlp)
    hidden_states = siglip_output(intermediate, attention_output, parameters=parameters.mlp)
    return hidden_states


def siglip_layer(hidden_states, attention_mask, parameters):
    """Single SigLIP transformer layer."""
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.norm1.weight,
        bias=parameters.norm1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    multi_head_attention_output = siglip_attention(
        layernorm_before_output,
        attention_mask=attention_mask,
        parameters=parameters.attn,
    )

    multi_head_attention_output = ttnn.add(
        multi_head_attention_output, hidden_states,
        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.norm2.weight,
        bias=parameters.norm2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    feedforward_output = siglip_feedforward(
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )
    return feedforward_output


def siglip_encoder(embeddings, head_masks, parameters, layer_end_index=None):
    """SigLIP encoder - stack of transformer layers."""
    encoder_input = embeddings
    if layer_end_index is None:
        layer_end_index = len(parameters)

    params = [parameters[index] for index in parameters]
    encoder_output = None

    for index, param in enumerate(params[:layer_end_index]):
        encoder_output = siglip_layer(encoder_input, head_masks[index], param)
        encoder_input = encoder_output

    return encoder_output


# ============================================================================
# Helper Functions for Attention Operations
# ============================================================================

def _split_qkv_and_reshape_heads(qkv_tensor, num_heads):
    """
    Split combined QKV tensor and reshape for multi-head attention.
    """
    batch_size = qkv_tensor.shape[0]
    seq_len = qkv_tensor.shape[1]
    total_hidden_size = qkv_tensor.shape[2]
    hidden_size = total_hidden_size // 3
    head_size = hidden_size // num_heads
    
    # Create output template tensor for ttsim's split
    output_template = ttnn.Tensor(
        shape=(batch_size, seq_len, hidden_size),
        dtype=ttnn.bfloat16,
        device=qkv_tensor.device
    )
    
    result = ttnn.split(
        qkv_tensor,
        output_template,
        num_splits=3,
        num_outputs=3,
        dim=-1,
    )
    
    # Extract Q, K, V
    query = result[0]
    key = result[1]  
    value = result[2]
    
    # Continue with reshaping...
    query = ttnn.reshape(query, (batch_size, seq_len, num_heads, head_size))
    key = ttnn.reshape(key, (batch_size, seq_len, num_heads, head_size))
    value = ttnn.reshape(value, (batch_size, seq_len, num_heads, head_size))
    
    query = ttnn.permute(query, (0, 2, 1, 3))
    key = ttnn.permute(key, (0, 2, 1, 3))
    value = ttnn.permute(value, (0, 2, 1, 3))
    
    key = ttnn.permute(key, (0, 1, 3, 2))
    
    return query, key, value


def _concatenate_heads(context_layer, num_heads):
    """
    Concatenate attention heads back together.

    Input: [B, num_heads, S, head_size]
    Output: [B, S, hidden_size]
    """
    batch_size = context_layer.shape[0]
    seq_len = context_layer.shape[2]
    head_size = context_layer.shape[3]
    hidden_size = num_heads * head_size

    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.reshape(context_layer, (batch_size, seq_len, hidden_size))

    return context_layer


def _select_index(tensor, dim, index):
    """
    Select a single index along a dimension.

    For dim=1, index=0: selects CLS token
    Input: [B, S, H]
    Output: [B, H]
    """
    shape = list(tensor.shape)
    batch_size = shape[0]
    hidden_size = shape[-1]

    if dim == 1 and index == 0:
        seq_len = shape[1]
        flat = ttnn.reshape(tensor, (batch_size, seq_len * hidden_size))
        # Keep old behavior for now; update later if this path fails in ttsim
        splits = ttnn.split(flat, seq_len, dim=-1)
        cls_output = splits[0]
        return cls_output

    raise NotImplementedError(f"_select_index not implemented for dim={dim}, index={index}")


# ============================================================================
# DINOv2 Transformer Blocks
# ============================================================================

def dinov2_embedding(var0, *args):
    """
    DINOv2 embedding layer.

    Args:
        var0: TTNN NHWC image tensor
        args[0]: projection weight
        args[1]: projection bias
        args[2]: positional embeddings
        args[3]: cls token
        args[4]: register tokens
    """
    var2 = vit_patch_embeddings_weight_vars(None, var0, args[0], args[1], patch_size=14)
    var5 = ttnn.add(var2, args[2])
    ttnn.deallocate(var2)

    cls_token = args[3]
    reg_tokens = args[4]

    cls_token = ttnn.to_layout(cls_token, ttnn.ROW_MAJOR_LAYOUT)
    reg_tokens = ttnn.to_layout(reg_tokens, ttnn.ROW_MAJOR_LAYOUT)
    var5 = ttnn.to_layout(var5, ttnn.ROW_MAJOR_LAYOUT)

    var6 = ttnn.concat([cls_token, reg_tokens, var5], dim=1)

    return var6


def dinov2_attention(var0, *args, num_heads=16):
    """
    DINOv2 attention block with layer scale.
    """
    *_, hidden_size = var0.shape
    head_size = hidden_size // num_heads

    hidden_states = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    query_key_value = ttnn.linear(
        hidden_states,
        args[3],
        bias=args[2],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.reallocate(hidden_states)

    query, key, value = _split_qkv_and_reshape_heads(query_key_value, num_heads=num_heads)
    ttnn.deallocate(query_key_value)

    scale = 1.0 / (head_size ** 0.5)
    query = ttnn.multiply(query, scale)
    value = ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query, key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attention_probs = ttnn.softmax(attention_scores)
    ttnn.deallocate(attention_scores)

    context_layer = ttnn.matmul(
        attention_probs, value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = _concatenate_heads(context_layer, num_heads=num_heads)

    self_output = ttnn.linear(
        context_layer,
        args[5],
        bias=args[4],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    ttnn.deallocate(context_layer)

    var19 = ttnn.multiply(self_output, args[6])
    var20 = ttnn.add(var0, var19)

    return var20


def dinov2_feedforward(var0, *args):
    """
    DINOv2 feedforward block with layer scale.
    """
    hidden_states = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    hidden_states = ttnn.linear(
        hidden_states,
        args[3],
        bias=args[2],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        activation="gelu",
    )

    hidden_states = ttnn.linear(
        hidden_states,
        args[5],
        bias=args[4],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    hidden_states = ttnn.multiply(hidden_states, args[6])
    var11 = ttnn.add(var0, hidden_states)

    return var11


def dinov2_head(var0, *args):
    """
    DINOv2 head - final layer norm and CLS token extraction.
    """
    var1 = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    cls_output = _select_index(var1, dim=1, index=0)
    return cls_output


# ============================================================================
# Weight Preprocessing Functions (Polaris - uses ttnn_from_numpy)
# ============================================================================

def upchannel_attn_weight_bias(qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads):
    """Pad attention weights for tile-friendly dimensions."""
    qkv = 3
    is_padding_required = (qkv_weight.shape[0] // (num_heads * qkv)) % 32 != 0

    if is_padding_required:
        padded_val = int(
            np.ceil(qkv_weight.shape[0] / (num_heads * qkv * 32)) * (num_heads * qkv * 32)
        )

        new_qkv_weight_4d = np.zeros(
            (padded_val, qkv_weight.shape[1]), dtype=qkv_weight.dtype
        ).reshape(qkv, num_heads, -1, qkv_weight.shape[1])
        reshaped_qkv_weight = qkv_weight.reshape(qkv, num_heads, -1, qkv_weight.shape[1])
        new_qkv_weight_4d[:, :, :reshaped_qkv_weight.shape[2], :] = reshaped_qkv_weight
        new_qkv_weight_2d = new_qkv_weight_4d.reshape(padded_val, qkv_weight.shape[1])

        new_qkv_bias_3d = np.zeros(
            (padded_val,), dtype=qkv_bias.dtype
        ).reshape(qkv, num_heads, -1)
        reshaped_qkv_bias = qkv_bias.reshape(qkv, num_heads, -1)
        new_qkv_bias_3d[:, :, :reshaped_qkv_bias.shape[2]] = reshaped_qkv_bias
        new_qkv_bias_1d = new_qkv_bias_3d.reshape((-1,))

        new_proj_weight_3d = np.zeros(
            (proj_weight.shape[0], padded_val // qkv), dtype=proj_weight.dtype
        ).reshape(proj_weight.shape[0], num_heads, -1)
        reshaped_proj_head = proj_weight.reshape(proj_weight.shape[0], num_heads, -1)
        new_proj_weight_3d[:, :, :reshaped_proj_head.shape[2]] = reshaped_proj_head
        new_proj_weight_2d = new_proj_weight_3d.reshape(
            (proj_weight.shape[0], padded_val // qkv)
        )

        qkv_weight = new_qkv_weight_2d
        qkv_bias = new_qkv_bias_1d
        proj_weight = new_proj_weight_2d

    return qkv_weight, qkv_bias, proj_weight, proj_bias


def prepare_dinov2_embedding_constants(tensors, device):
    """Preprocess DINOv2 embedding weights for TTNN."""
    assert len(tensors) == 2
    proj_weight = tensors[0]
    proj_bias = tensors[1]

    three_times_hidden_size, c, _, _ = proj_weight.shape
    pad_value = 4 - c

    preprocessed_weight = np.pad(
        proj_weight, ((0, 0), (0, pad_value), (0, 0), (0, 0)), mode='constant'
    )
    preprocessed_weight = np.transpose(preprocessed_weight, (2, 3, 1, 0))
    preprocessed_weight = preprocessed_weight.reshape(-1, three_times_hidden_size)

    tensors[0] = ttnn_from_numpy(
        preprocessed_weight.astype(np.float32),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tensors[1] = ttnn_from_numpy(
        np.expand_dims(proj_bias, 0).astype(np.float32),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    return [tensors[0], tensors[1]]


def prepare_dinov2_attention_constants(tensors, device):
    """Preprocess DINOv2 attention weights for TTNN."""
    assert len(tensors) == 7

    for i in range(7):
        arr = cast(npt.NDArray[Any], tensors[i])
        if arr.ndim == 1:
            arr_for_tensor = cast(npt.NDArray[Any], np.expand_dims(arr, 0))
        else:
            arr_for_tensor = arr

        tensors[i] = ttnn_from_numpy(
            np.ascontiguousarray(arr_for_tensor).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    return tensors


def prepare_dinov2_feedforward_constants(tensors, device):
    """Preprocess DINOv2 feedforward weights for TTNN."""
    assert len(tensors) == 7

    for i in range(7):
        arr = cast(npt.NDArray[Any], tensors[i])
        if arr.ndim == 1:
            arr_for_tensor = cast(npt.NDArray[Any], np.expand_dims(arr, 0))
        else:
            arr_for_tensor = arr

        tensors[i] = ttnn_from_numpy(
            np.ascontiguousarray(arr_for_tensor).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    return tensors


def prepare_dinov2_head_constants(tensors, device):
    """Preprocess DINOv2 head weights for TTNN."""
    assert len(tensors) == 2

    for i in range(2):
        arr = cast(npt.NDArray[Any], tensors[i])
        if arr.ndim == 1:
            arr_for_tensor = cast(npt.NDArray[Any], np.expand_dims(arr, 0))
        else:
            arr_for_tensor = arr

        tensors[i] = ttnn_from_numpy(
            np.ascontiguousarray(arr_for_tensor).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    return tensors


def get_dinov2_params_from_numpy(state_dict, num_blocks):
    """Extract DINOv2 parameters from numpy state dict."""
    return {
        "embeddings": [
            state_dict["patch_embed.proj.weight"],
            state_dict["patch_embed.proj.bias"],
            state_dict["pos_embed"],
            state_dict["cls_token"],
            state_dict["reg_token"],
        ],
        "encoder": {
            f"layer{i}": {
                "attention": [
                    state_dict[f"blocks.{i}.norm1.weight"],
                    state_dict[f"blocks.{i}.norm1.bias"],
                    state_dict[f"blocks.{i}.attn.qkv.bias"],
                    state_dict[f"blocks.{i}.attn.qkv.weight"].T,
                    state_dict[f"blocks.{i}.attn.proj.bias"],
                    state_dict[f"blocks.{i}.attn.proj.weight"].T,
                    state_dict[f"blocks.{i}.ls1.scale_factor"],
                ],
                "feed_forward": [
                    state_dict[f"blocks.{i}.norm2.weight"],
                    state_dict[f"blocks.{i}.norm2.bias"],
                    state_dict[f"blocks.{i}.mlp.fc1.bias"],
                    state_dict[f"blocks.{i}.mlp.fc1.weight"].T,
                    state_dict[f"blocks.{i}.mlp.fc2.bias"],
                    state_dict[f"blocks.{i}.mlp.fc2.weight"].T,
                    state_dict[f"blocks.{i}.ls2.scale_factor"],
                ],
            }
            for i in range(num_blocks)
        },
    }


def dinov2_encoder(state_dict, num_blocks, ttnn_device, num_output_layers=None):
    """
    Create DINOv2 encoder from numpy state dict.
    """
    parameters = get_dinov2_params_from_numpy(state_dict, num_blocks)

    if num_output_layers is None:
        num_output_layers = num_blocks - 1

    embedding_params = prepare_dinov2_embedding_constants(
        parameters["embeddings"][:2], ttnn_device
    )

    pos_embed = ttnn_from_numpy(
        parameters["embeddings"][2].astype(np.float32),
        dtype=ttnn.bfloat16, device=ttnn_device
    )
    cls_token = ttnn_from_numpy(
        parameters["embeddings"][3].astype(np.float32),
        dtype=ttnn.bfloat16, device=ttnn_device
    )
    reg_token = ttnn_from_numpy(
        parameters["embeddings"][4].astype(np.float32),
        dtype=ttnn.bfloat16, device=ttnn_device
    )

    parameters["embeddings"] = embedding_params + [pos_embed, cls_token, reg_token]

    def get_layer_num(key):
        if isinstance(key, int):
            return key
        if isinstance(key, str) and key.startswith("layer"):
            return int(key.replace("layer", ""))
        return int(key)

    encoder_layers = sorted(parameters["encoder"].keys(), key=get_layer_num)

    for idx, layer in enumerate(encoder_layers):
        if idx >= num_output_layers:
            continue

        attention_params = prepare_dinov2_attention_constants(
            parameters["encoder"][layer]["attention"][:7], ttnn_device
        )
        parameters["encoder"][layer]["attention"] = attention_params

        feedforward_params = prepare_dinov2_feedforward_constants(
            parameters["encoder"][layer]["feed_forward"][:7], ttnn_device
        )
        parameters["encoder"][layer]["feed_forward"] = feedforward_params

    def model_forward(pixel_values):
        """Forward pass through DINOv2 encoder."""
        embeddings_output = dinov2_embedding(pixel_values, *parameters["embeddings"])
        embeddings_output = ttnn.to_layout(embeddings_output, layout=ttnn.TILE_LAYOUT)

        for idx, layer in enumerate(encoder_layers):
            if idx >= num_output_layers:
                break
            embeddings_output = dinov2_attention(
                embeddings_output, *parameters["encoder"][layer]["attention"]
            )
            embeddings_output = dinov2_feedforward(
                embeddings_output, *parameters["encoder"][layer]["feed_forward"]
            )

        return embeddings_output

    return model_forward


# ============================================================================
# Top-level Featurizer
# ============================================================================

def ttnn_featurizer(embedding, encoder, pixel):
    """Run embedding + encoder with TTNN ops."""
    embd = embedding(pixel)
    embd = ttnn.to_layout(embd, layout=ttnn.TILE_LAYOUT)
    return encoder(embd)