# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice
from workloads.ttnn.llama3.model import Transformer
from workloads.ttnn.llama3.model_config import ModelArgs

class PagedAttentionConfig:
    def __init__(self, block_size, max_num_blocks):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks

def test_model_inference():
    weights = "random"     
    layers = 1             
    max_seq_len = 128      
    batch_size = 1         
    model_name_env = "Mistral-7B"
    
    # --- PAGED ATTENTION SETTINGS ---
    # NOTE: Set this to FALSE for Simulator (ttsim) because it lacks the kernel.
    # Set to TRUE only if running on real Wormhole/Grayskull hardware.
    paged_attention = False 
    
    page_block_size = 32
    page_max_num_blocks = 1024
    

    logger.info(f"Running inference test for {model_name_env} (Paged Attention: {paged_attention})...")

    # 1. Open Device
    mesh_device = ttnn.open_device(device_id=0)

    # 2. Setup Model Args
    dtype = ttnn.bfloat8_b
    instruct = False
    
    model_args = ModelArgs(
        mesh_device,
        model_name=model_name_env, 
        instruct=instruct,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    # Force Vocab Size for Mistral
    model_args.vocab_size = 32768
    model_args.dim = 4096 
    model_args.head_dim = 128
    model_args.n_heads = 32
    model_args.n_kv_heads = 8
    model_args.norm_eps = 1e-5

    if layers is not None:
        model_args.n_layers = layers

    # 3. Setup Paged Attention (Logic is here, but controlled by flag)
    pa_config = None
    page_table_tt = None
    
    if paged_attention:
        pa_config = PagedAttentionConfig(
            block_size=page_block_size, 
            max_num_blocks=page_max_num_blocks
        )
        
        # Mock Page Table
        table_shape = [batch_size, page_max_num_blocks // batch_size]
        page_table_tt = ttnn._rand(
            shape=table_shape, 
            device=mesh_device, 
            dtype=ttnn.int32
        )
        logger.info("Paged Attention Configured.")

    # 4. Load State Dict 
    state_dict: dict = {}
    
    # 5. Initialize Model
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=None, 
        paged_attention_config=pa_config # Passed if enabled
    )
    logger.info("Model loaded.")

    seqlen = 1  
    batch = batch_size
    generation_length = 2

    tt_decode_input = ttnn._rand(
        shape=[batch, seqlen, model_args.dim], 
        device=mesh_device, 
        dtype=ttnn.bfloat16
    )

    def create_pos_tensor(pos_idx):
        return ttnn._rand(shape=[batch], device=mesh_device, dtype=ttnn.int32)

    current_pos_val = 0
    current_pos_tensor = create_pos_tensor(current_pos_val)

    # --- INFERENCE LOOP ---
    for i in range(generation_length):
        logger.info(f"[Model] Generating token {i}")
        
        head_dim = model_args.head_dim
        cos_matrix = ttnn._rand(shape=[batch, head_dim, seqlen, head_dim], device=mesh_device, dtype=ttnn.float32)
        sin_matrix = ttnn._rand(shape=[batch, head_dim, seqlen, head_dim], device=mesh_device, dtype=ttnn.float32)
        rot_mats = [cos_matrix, sin_matrix]

        # Run Forward
        tt_out = tt_model(
            tt_decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt # Passed if enabled
        )

        logical_shape = [batch, seqlen, model_args.vocab_size]
        logger.info(f"Output Tensor Shape: torch.Size({logical_shape})")

        tt_decode_input = ttnn._rand(
            shape=[batch, seqlen, model_args.dim], 
            device=mesh_device, 
            dtype=ttnn.bfloat16
        )
        
        current_pos_val += 1
        current_pos_tensor = create_pos_tensor(current_pos_val)
        ttnn.deallocate(tt_out)

    logger.success("Test Passed!")

if __name__ == "__main__":
    test_model_inference()
