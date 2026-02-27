#!/usr/bin/env python

# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.tensor import DataType
from workloads.ttnn.tt_transformers.decoder import TransformerBlock
from workloads.ttnn.tt_transformers.rope import RotarySetup
from workloads.ttnn.tt_transformers.model_config import ModelArgs


def test_decoder_inference():
    mesh_device = ttnn.open_device(device_id=0)

    max_seq_len = 256
    paged_attention = False
    page_params = [{"page_block_size": 32, "page_max_num_blocks": 1024}]
    dtype = ttnn.bfloat8_b
    batch_size = 1

    model_args = ModelArgs(
        mesh_device,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    model_args.n_layers = 1

    state_dict = None  # model_args.load_state_dict()
    generation_length = 1

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Prepare page table for paged attention
    page_table_tt = None
    paged_attention_config = None

    # Initialize TT model
    tt_model = TransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
    )

    seqlen = 1

    # Initial positions
    current_pos = ttnn.Tensor(
        shape=(batch_size,),
        device=mesh_device,
        dtype=ttnn.int32,
    )

    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0)
            if (model_args.is_galaxy and batch_size > 1)
            else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        pt_decode_input = ttnn._rand(
            shape=(seqlen, batch_size, model_args.dim),
            device=mesh_device,
            dtype=dtype,
        )

        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            None,  # model_args.model_config["DECODE_RESIDUAL_MEMCFG"]
        )

        # Get cos/sin matrices for current position
        current_pos = current_pos.unsqueeze(0)
        rot_mats = rope_setup.get_rot_mats(current_pos)

        logger.info(f"decode input shape: {decode_input.shape}")

        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )

        if tt_out.shape == decode_input.shape:
            logger.success(f"Decoder Block Passed for token {i}!")
        else:
            logger.error(f"Decoder Block Failed! Expected shape: {decode_input.shape}, but got: {tt_out.shape}")

        # Reset position for next token
        current_pos = ttnn.Tensor(
            shape=(batch_size,),
            device=mesh_device,
            dtype=ttnn.int32,
        )

        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0)
                if (model_args.is_galaxy and batch_size > 1)
                else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )


if __name__ == "__main__":
    test_decoder_inference()
    logger.info("All tests completed!")