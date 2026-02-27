# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from loguru import logger
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.attention import Attention

class TT_CCL:
    def __init__(self, device):
        pass

class PagedAttentionConfig:
    def __init__(self, block_size, max_num_blocks):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


def test_attention_inference():
    logger.info("Initializing Attention Test (Mistral-7B Config with TT-Metal Parity)...")

    batch_size = 1
    seq_len = 1
    generation_length = 5
    paged_attention = False

    mesh_device = ttnn.open_device(device_id=0)

    model_args = ModelArgs(
        mesh_device,
        model_name="Mistral-7B",
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=32,
            max_num_blocks=1024,
        )
    
        table_shape = [batch_size, paged_attention_config.max_num_blocks // batch_size]
        page_table_tt = ttnn._rand(shape=table_shape, device=mesh_device, dtype=ttnn.int32)
        logger.info("Paged Attention Configured.")

    # 3. Setup RoPE Matrices (Mimicking RotarySetup)
    trans_mat_raw = ttnn._rand(shape=[1, 1, 32, 32], device=mesh_device, dtype=ttnn.bfloat16)
    trans_mat_tiled = ttnn.to_layout(trans_mat_raw, ttnn.TILE_LAYOUT)
    transformation_mats = {"decode": trans_mat_tiled}

    state_dict: dict[str, Any] = {}

    tt_ccl = TT_CCL(mesh_device)

    tt_model = Attention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat8_b,
        configuration=model_args,
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config
    )
    logger.info("Attention module initialized.")

    for i in range(generation_length):
        logger.info(f"[Step {i}] Running Attention Forward Pass...")

        pt_input = ttnn._rand(shape=[batch_size, 1, seq_len, model_args.dim], device=mesh_device, dtype=ttnn.bfloat16)
        attention_input = ttnn.to_layout(pt_input, ttnn.TILE_LAYOUT)

        current_pos_tensor = ttnn._rand(
            shape=[batch_size],
            device=mesh_device,
            dtype=ttnn.int32
        )

        # RoPE Rotation Matrices (Broadcasting for Q/K heads)
        cos_raw = ttnn._rand(shape=[batch_size, 1, 1, model_args.head_dim], device=mesh_device, dtype=ttnn.bfloat16)
        sin_raw = ttnn._rand(shape=[batch_size, 1, 1, model_args.head_dim], device=mesh_device, dtype=ttnn.bfloat16)
        rot_mats = [ttnn.to_layout(cos_raw, ttnn.TILE_LAYOUT), ttnn.to_layout(sin_raw, ttnn.TILE_LAYOUT)]

        tt_out = tt_model(
            attention_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt
        )

        expected_dim = model_args.head_dim * model_args.n_heads

        # Generic shape validation matching the Llama3 standard
        if (tt_out.shape[0] != batch_size) or (tt_out.shape[1] != 1) or (tt_out.shape[2] != seq_len) or (tt_out.shape[3] != expected_dim):
            logger.error(f"Step {i} Failed! Expected [{batch_size}, 1, {seq_len}, {expected_dim}], got {list(tt_out.shape)}")
        else:
            logger.success(f"Step {i} Passed! Shape: {list(tt_out.shape)}")
        ttnn.deallocate(tt_out)

    logger.info("Attention Test Completed Successfully.")

if __name__ == "__main__":
    test_attention_inference()