# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.decoder import TransformerBlock
from ttsim.front.ttnn.device import Device as TTNNDevice

def test_model_full_prefill_qwen(wlname: str, mesh_device: TTNNDevice, cfg: dict):
    batch_size = 1 # only support batch size 1 for prefill
    seqlen = 128
    dtype = ttnn.bfloat16
    model_name = cfg.get('model_name', 'Qwen2.5-7B')

    model_args = ModelArgs(
        mesh_device,
        model_name=model_name,
        max_batch_size=batch_size,
        max_seq_len=seqlen,
        instruct=False
    )

    dim = model_args.dim
    head_dim = model_args.head_dim
    vocab_size = model_args.vocab_size

    logger.info(f"Running Full Model {wlname} test for {model_name} (Dim: {dim})...")

    logger.info("Constructing explicit Decoder + LM Head for Simulator...")
    trans_mat_raw = ttnn.Tensor(shape=[1, 1, 32, 32], device=mesh_device, dtype=ttnn.bfloat16)
    transformation_mats = {"prefill": trans_mat_raw}

    layer = TransformerBlock(
        mesh_device=mesh_device,
        state_dict={},
        weight_cache_path=None,
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        args=model_args,
        paged_attention_config=None
    )

    logger.info("Injecting dummy norm weights...")
    dummy_norm_weight = ttnn.Tensor(shape=[1, 1, 1, dim], device=mesh_device, dtype=ttnn.bfloat16)
    layer.attention_norm.weight = dummy_norm_weight
    layer.ff_norm.weight = dummy_norm_weight

    qwen_lm_head = ttnn.Tensor(shape=[1, 1, dim, vocab_size], device=mesh_device, dtype=ttnn.bfloat16)
    tt_prefill_input = ttnn.Tensor(shape=[batch_size, 1, seqlen, dim], device=mesh_device, dtype=ttnn.bfloat16)

    cos_matrix = ttnn.Tensor(shape=[batch_size, 1, seqlen, head_dim], device=mesh_device, dtype=ttnn.float32)
    sin_matrix = ttnn.Tensor(shape=[batch_size, 1, seqlen, head_dim], device=mesh_device, dtype=ttnn.float32)
    rot_mats = [cos_matrix, sin_matrix]

    logger.info("[Model] Running Context Prefill Pass...")

    hidden_states = layer(
        tt_prefill_input,
        current_pos=None,
        rot_mats=rot_mats,
        mode="prefill",
        page_table=None
    )

    tt_out = ttnn.linear(hidden_states, qwen_lm_head)
    actual_shape = list(tt_out.shape)

    logger.info(f"Actual Simulator Output Shape: {actual_shape}")

    if actual_shape[-1] == vocab_size:
        logger.success(f"Prefill Test Passed! Output perfectly matches Qwen {vocab_size} Vocab.")
    else:
        logger.error(f"Prefill Test Failed! Expected {vocab_size}, got {actual_shape[-1]}")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_model_full_prefill_qwen(wlname="qwen_full_prefill", mesh_device=mesh_device, cfg={'model_name': 'Qwen2.5-7B'})
    ttnn.close_device(mesh_device)
