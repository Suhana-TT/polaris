# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
from loguru import logger
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.attention import Attention

class TT_CCL:
    def __init__(self, device):
        pass


class PagedAttentionConfig:
    def __init__(self, block_size, max_num_blocks):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks

def test_attention_prefill():
    logger.info("Initializing Attention PREFILL Test (Mistral-7B Config)...")
    batch_size = 1
    seq_len = 128

    mesh_device = ttnn.open_device(device_id=0)

    model_args = ModelArgs(
        mesh_device,
        model_name="Mistral-7B",
        max_batch_size=batch_size,
        max_seq_len=seq_len,
    )

    state_dict: dict[str, Any] = {}
    # --- Transformation Matrix MUST be 32x32 ---
    trans_mat_raw = ttnn._rand(
        shape=[1, 1, 32, 32],
        device=mesh_device,
        dtype=ttnn.bfloat16
    )
    trans_mat_tiled = ttnn.to_layout(trans_mat_raw, ttnn.TILE_LAYOUT)

    # Optional: If L1 memory config causes simulator issues, you can remove the memory_config wrap
    # and just pass the tiled layout directly, but keeping it is fine if it works.
    trans_mat_l1 = ttnn.to_memory_config(trans_mat_tiled, ttnn.L1_MEMORY_CONFIG)

    transformation_mats = {
        "prefill": trans_mat_l1
    }

    tt_model = Attention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat8_b,
        configuration=model_args,
        transformation_mats=transformation_mats,
        paged_attention_config=None
    )
    logger.info("Attention module initialized.")

    def get_random_input():
        t = ttnn._rand(
            shape=[batch_size, 1, seq_len, model_args.dim],
            device=mesh_device,
            dtype=ttnn.bfloat16
        )
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    logger.info(f"Running Attention PREFILL Forward Pass (Seq Len: {seq_len})...")

    attention_input = get_random_input()
    current_pos_tensor = None

    cos_raw = ttnn._rand(
        shape=[batch_size, 1, seq_len, model_args.head_dim],
        device=mesh_device,
        dtype=ttnn.bfloat16
    )
    sin_raw = ttnn._rand(
        shape=[batch_size, 1, seq_len, model_args.head_dim],
        device=mesh_device,
        dtype=ttnn.bfloat16
    )

    cos_tiled = ttnn.to_layout(cos_raw, ttnn.TILE_LAYOUT)
    sin_tiled = ttnn.to_layout(sin_raw, ttnn.TILE_LAYOUT)
    rot_mats = [cos_tiled, sin_tiled]

    # Forward Pass
    tt_out = tt_model(
        attention_input,
        current_pos=current_pos_tensor,
        rot_mats=rot_mats,
        user_id=0, # Added for parity with TT-Metal
        mode="prefill",
        page_table=None
    )

    actual_shape = list(tt_out.shape)
    logger.info(f"Output Shape: {actual_shape}")

    # Dynamically set expected dimensions from model_args (Removed hardcoded 4096)
    expected_dim = model_args.dim
    expected_seq = seq_len

    # Generic shape validation for all 4 dimensions (matches Llama3 standard)
    if (tt_out.shape[0] != batch_size) or (tt_out.shape[1] != 1) or (tt_out.shape[2] != expected_seq) or (tt_out.shape[3] != expected_dim):
        logger.error(f"Failed! Expected [{batch_size}, 1, {expected_seq}, {expected_dim}], got {actual_shape}")
    else:
        logger.success(f"Passed! Shape: {actual_shape}")

    ttnn.deallocate(tt_out)
    logger.info("Done.")

if __name__ == "__main__":
    test_attention_prefill()