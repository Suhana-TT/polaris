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

class PagedAttentionConfig:
    def __init__(self, block_size, max_num_blocks):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks

class Generator:
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device

    def prefill_forward_text(self, input_tensor, page_table=None, kv_cache=None, prompt_lens=None):
        logger.info(f"Generator: Executing prefill_forward_text for seq_len {prompt_lens[0]}")
        
        seq_len = prompt_lens[0]
        # Use ttnn._rand or similar for simulator compatibility
        cos_raw = ttnn._rand(shape=[1, 1, seq_len, self.args.head_dim], device=self.device, dtype=ttnn.bfloat16)
        sin_raw = ttnn._rand(shape=[1, 1, seq_len, self.args.head_dim], device=self.device, dtype=ttnn.bfloat16)
        rot_mats = [cos_raw, sin_raw]
        
        return self.model(
            input_tensor,
            current_pos=None,
            rot_mats=rot_mats,
            mode="prefill",
            page_table=page_table
        )


def test_model_paged_prefill(wlname: str, mesh_device: TTNNDevice, cfg: dict):
    logger.info(f"Starting {wlname} Model Inference (Paged Attention Prefill)...")

    # 1. Setup constants
    batch_size = 1  # For prefill we only support batch_size = 1
    seq_len = 128
    model_name = cfg.get('model_name', 'Mistral-7B')

    # 2. Setup Model Args (PASS THE NAME HERE!)
    # This automatically sets dim=4096, n_heads=32, etc.
    model_args = ModelArgs(mesh_device, model_name=model_name)

    # Only set things that are NOT in the default config
    model_args.max_batch_size = batch_size
    model_args.instruct = False
    model_args.max_seq_len = 2048

    # 3. Setup Paged Attention Config
    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

    # 4. Create Page Table Dummy Tensor
    page_table = ttnn._rand(shape=[batch_size, 1024], device=mesh_device, dtype=ttnn.int32)

    # 5. Initialize TT Model
    trans_mat_raw = ttnn._rand(shape=[1, 1, 32, 32], device=mesh_device, dtype=ttnn.bfloat16)
    transformation_mats = {"prefill": trans_mat_raw}

    logger.info(f"Initializing TransformerBlock (Dim: {model_args.dim})...")
    tt_model = TransformerBlock(
        mesh_device=mesh_device,
        state_dict={},
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat16,
        transformation_mats=transformation_mats,
        args=model_args,
        paged_attention_config=paged_attention_config
    )

    # 6. Initialize Generator
    generator = Generator(tt_model, model_args, mesh_device)

    # 7. Prepare Input (Use model_args.dim!)
    tt_prefill_input = ttnn._rand(shape=[batch_size, 1, seq_len, model_args.dim], device=mesh_device, dtype=ttnn.bfloat16)

    # 8. Run TT Model
    logger.info("Running generator.prefill_forward_text...")
    tt_output = generator.prefill_forward_text(
        tt_prefill_input,
        page_table=page_table,
        kv_cache=None,
        prompt_lens=[seq_len]
    )

    actual_shape = list(tt_output.shape)
    logger.info(f"Output Shape: {actual_shape}")

    # 9. Validate against model_args.dim
    if actual_shape[-1] == model_args.dim:
        logger.success(f"test_model_inference (Paged Attention) Passed! Matches {model_args.dim}")
    else:
        logger.error(f"Failed! Expected {model_args.dim}, got {actual_shape[-1]}")

    logger.info("Done.")

if __name__ == "__main__":
    ttnn_device = ttnn.open_device(device_id=0)
    test_model_paged_prefill("mistral_prefill", ttnn_device, cfg={'model_name': 'Mistral-7B'})
    ttnn.close_device(ttnn_device)
