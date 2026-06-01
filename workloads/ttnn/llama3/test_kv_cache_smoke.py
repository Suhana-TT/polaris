#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn
import workloads.ttnn.tt_transformers.utils as utils
from workloads.ttnn.tt_transformers.attention import Attention
from workloads.ttnn.tt_transformers.rope import RotarySetup
from workloads.ttnn.tt_transformers.model_config import ModelArgs


def main():
    logger.info("Setting up model args and attention module")

    mesh_device = ttnn.open_device(device_id=0)

    try:
        batch_size = 1
        seq_len = 128
        max_seq_len = 256
        decode_seq_len = 1
        dtype = ttnn.bfloat8_b

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        model_args.n_layers = 1

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

        tt_model = Attention(
            mesh_device=mesh_device,
            state_dict={},
            weight_cache_path=None,
            layer_num=0,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=model_args,
            paged_attention_config=None,
            use_paged_kv_cache=False,
        )

        call_counts = {
            "fill_cache": 0,
            "paged_fill_cache": 0,
            "paged_update_cache": 0,
        }

        orig_fill_cache = utils.fill_cache
        orig_paged_fill_cache = utils.paged_fill_cache
        orig_paged_update_cache = utils.paged_update_cache

        def wrapped_fill_cache(*args, **kwargs):
            call_counts["fill_cache"] += 1
            logger.info(f"[HOOK] utils.fill_cache called ({call_counts['fill_cache']})")
            return orig_fill_cache(*args, **kwargs)

        def wrapped_paged_fill_cache(*args, **kwargs):
            call_counts["paged_fill_cache"] += 1
            logger.info(f"[HOOK] utils.paged_fill_cache called ({call_counts['paged_fill_cache']})")
            return orig_paged_fill_cache(*args, **kwargs)

        def wrapped_paged_update_cache(*args, **kwargs):
            call_counts["paged_update_cache"] += 1
            logger.info(f"[HOOK] utils.paged_update_cache called ({call_counts['paged_update_cache']})")
            return orig_paged_update_cache(*args, **kwargs)

        utils.fill_cache = wrapped_fill_cache
        utils.paged_fill_cache = wrapped_paged_fill_cache
        utils.paged_update_cache = wrapped_paged_update_cache

        logger.info("Running PREFILL")

        prefill_input_raw = ttnn._rand(
            shape=[batch_size, 1, seq_len, model_args.dim],
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        prefill_input = ttnn.to_layout(prefill_input_raw, ttnn.TILE_LAYOUT)

        rot_mats_prefill = [
            ttnn._rand([1, 1, seq_len, model_args.head_dim], device=mesh_device, dtype=ttnn.bfloat16),
            ttnn._rand([1, 1, seq_len, model_args.head_dim], device=mesh_device, dtype=ttnn.bfloat16),
        ]

        prefill_out = tt_model(
            prefill_input,
            current_pos=None,
            rot_mats=rot_mats_prefill,
            mode="prefill",
            page_table=None,
        )

        logger.info(f"Prefill output shape: {list(prefill_out.shape)}")
        logger.info(f"Prefill cache-call counts: {call_counts}")

        assert call_counts["fill_cache"] == 2, (
            f"Expected 2 fill_cache calls in prefill (K and V), got {call_counts['fill_cache']}"
        )
        assert call_counts["paged_fill_cache"] == 0, (
            f"Expected 0 paged_fill_cache calls in non-paged prefill, got {call_counts['paged_fill_cache']}"
        )
        assert call_counts["paged_update_cache"] == 0, (
            f"Expected 0 paged_update_cache calls before decode, got {call_counts['paged_update_cache']}"
        )

        logger.info("Running DECODE")

        decode_input_raw = ttnn._rand(
            shape=(decode_seq_len, batch_size, model_args.dim),
            device=mesh_device,
            dtype=dtype,
        )
        decode_input_clone = decode_input_raw.clone()
        decode_input = model_args.prepare_residual_tensor_decode(
            decode_input_clone,
            None,
        )

        current_pos_tensor = ttnn.Tensor(
            shape=(batch_size,),
            device=mesh_device,
            dtype=ttnn.int32,
            data=np.array([seq_len], dtype=np.int32),
        )

        rot_mats_decode = [
            ttnn._rand([1, batch_size, 1, model_args.head_dim], device=mesh_device, dtype=ttnn.bfloat16),
            ttnn._rand([1, batch_size, 1, model_args.head_dim], device=mesh_device, dtype=ttnn.bfloat16),
        ]

        decode_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats_decode,
            mode="decode",
            page_table=None,
        )

        logger.info(f"Decode output shape: {list(decode_out.shape)}")
        logger.info(f"Final cache-call counts: {call_counts}")

        assert call_counts["paged_update_cache"] == 2, (
            f"Expected 2 paged_update_cache calls in decode (K and V), got {call_counts['paged_update_cache']}"
        )

        assert list(decode_out.shape) == list(decode_input.shape), (
            f"Decode output shape mismatch. Expected {list(decode_input.shape)}, got {list(decode_out.shape)}"
        )

        logger.success("KV cache smoke test PASSED")
        logger.success("Prefill filled cache and decode updated cache")

    finally:
        try:
            utils.fill_cache = orig_fill_cache
        except Exception:
            pass

        try:
            utils.paged_fill_cache = orig_paged_fill_cache
        except Exception:
            pass

        try:
            utils.paged_update_cache = orig_paged_update_cache
        except Exception:
            pass

        ttnn.close_device(mesh_device)


if __name__ == "__main__":
    main()