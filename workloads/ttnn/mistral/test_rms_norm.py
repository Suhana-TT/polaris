# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
from loguru import logger

# 1. SETUP PATH
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# 2. IMPORT SIMULATOR & WORKLOADS
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

# Import specific components for RMS Norm
from workloads.ttnn.llama3.model_config import ModelArgs
from workloads.ttnn.llama3.model import RMSNorm  # Assuming RMSNorm is available here or in common
# If TT_CCL or DistributedNorm are missing in your sim environment, 
# you might need to test RMSNorm directly. Assuming they exist for now:
try:
    from workloads.ttnn.llama3.ccl import TT_CCL 
    from workloads.ttnn.llama3.distributed_norm import DistributedNorm
except ImportError:
    # Fallback mock classes if specific files are missing in your local setup
    logger.warning("CCL/DistributedNorm not found, defining mocks for single-device test.")
    class TT_CCL:
        def __init__(self, device): pass
    class DistributedNorm:
        def __init__(self, norm, args, ccl, TG): self.norm = norm
        def __call__(self, x, mode): return self.norm(x)

def test_rms_norm_inference():
    logger.info("Initializing RMS Norm Test (Mistral-7B Config)...")

    # --- CONFIGURATION ---
    # We test 'decode' mode as it is the most common critical path
    mode = "decode" 
    batch_size = 1
    seq_len = 32  # Matching the torch.rand(1, 1, 32, dim) from your snippet
    
    # 1. Open Device
    mesh_device = ttnn.open_device(device_id=0)

    # 2. Setup Model Args (Mistral 7B)
    model_args = ModelArgs(
        mesh_device,
        model_name="Mistral-7B",
        max_batch_size=batch_size,
        max_seq_len=128,
    )

    # --- CRITICAL OVERRIDES ---
    model_args.dim = 4096 
    model_args.head_dim = 128
    model_args.n_heads = 32
    model_args.n_kv_heads = 8
    model_args.vocab_size = 32768
    model_args.norm_eps = 1e-5
    model_args.n_layers = 1
    # --------------------------

    # 3. Setup Components
    # We pass an empty state_dict. The model classes should handle initialization (random weights).
    state_dict = {}
    state_dict_prefix = "layers.0.attention_norm." # Mock prefix
    
    tt_ccl = TT_CCL(mesh_device)
    
    # Initialize Inner RMSNorm
    # Note: We use the config getters from model_args just like the original code
    tt_inner_norm = RMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key="weight", # Simplified key
        weight_dtype=ttnn.bfloat16,
        add_unit_offset=False, # Standard for Llama/Mistral (usually False)
        # is_distributed=False, # Simplify to False for single device sim
        # sharded_program_config=model_args.get_model_config().get("SHARDED_NORM_ATTN_PRGM_CFG"),
        # sharded_output_config=model_args.get_model_config().get("SHARDED_ATTN_INPUT_MEMCFG"),
        # tt_ccl=tt_ccl,
    )

    # Wrap in DistributedNorm (Or use Mock)
    tt_model = DistributedNorm(
        tt_inner_norm, 
        model_args, 
        tt_ccl, 
        TG=False
    )

    # 4. Create Input (Mocking torch.rand(1, 1, 32, dim))
    # Using ttnn._rand directly on device
    tt_input = ttnn._rand(
        shape=[batch_size, 1, seq_len, model_args.dim],
        device=mesh_device,
        dtype=ttnn.bfloat16
    )
    
    logger.info(f"Input Shape: {tt_input.shape}")

    # 5. Run Forward Pass
    logger.info(f"Running RMSNorm in {mode} mode...")
    
    tt_output = tt_model(tt_input, mode=mode)

    # 6. Validate Output (Shape Only)
    # The output should maintain the same shape as input [1, 1, 32, 4096]
    actual_shape = list(tt_output.shape)
    
    logger.info(f"Output Shape: {actual_shape}")

    # Validation Logic
    # We expect the last dim to be 4096 (Mistral Dim)
    # We expect the sequence dim to be 32
   # Validation Logic
    # Using [-1] and [-2] is safer as it ignores potential leading padding dims
    if actual_shape[-1] == 4096 and actual_shape[-2] == 32:
        logger.success(f"Test Passed! Output dimensions match Mistral (Seq: 32, Dim: 4096).")
    else:
        logger.error(f"Test Failed! Expected [..., 32, 4096], but got {actual_shape}")
        
    ttnn.deallocate(tt_output)
    logger.info("Done.")

if __name__ == "__main__":
    test_rms_norm_inference()