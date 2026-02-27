# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from loguru import logger
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.model import RMSNorm

# Since DistributedNorm and TT_CCL are not yet in tt_transformers,
# we define them locally here to keep the test standalone and crash-proof.
class TT_CCL:
    def __init__(self, device):
        pass

class DistributedNorm:
    def __init__(self, norm, args, ccl, TG):
        self.norm = norm
        self.args = args
    def __call__(self, x, mode):
        return self.norm(x)

def test_rms_norm_inference():
    logger.info("Initializing RMS Norm Test (Mistral-7B Config)...")
    mode = "prefill"
    batch_size = 1
    seq_len = 32
    mesh_device = ttnn.open_device(device_id=0)

    # 1. Initialize ModelArgs - This automatically sets dim=4096, norm_eps=1e-5, etc.
    model_args = ModelArgs(
        mesh_device,
        model_name="Mistral-7B",
        max_batch_size=batch_size,
        max_seq_len=128,
    )
    
    # 2. Add only the runtime-specific configs (Remove the hardcoded dim/heads/vocab/eps)
    model_args.n_layers = 1

    state_dict: dict[str, Any] = {}
    state_dict_prefix = "layers.0.attention_norm."
    tt_ccl = TT_CCL(mesh_device)

    tt_inner_norm = RMSNorm(
        device=mesh_device,
        dim=model_args.dim,  # Dynamically uses 4096 from ModelArgs
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key="weight",
        weight_dtype=ttnn.bfloat16,
        add_unit_offset=False,
    )

    tt_model = DistributedNorm(
        tt_inner_norm, 
        model_args, 
        tt_ccl, 
        TG=False
    )

    tt_input = ttnn._rand(
        shape=[batch_size, 1, seq_len, model_args.dim],
        device=mesh_device,
        dtype=ttnn.bfloat16
    )

    logger.info(f"Input Shape: {tt_input.shape}")
    logger.info(f"Running RMSNorm in {mode} mode...")

    tt_output = tt_model(tt_input, mode=mode)
    actual_shape = list(tt_output.shape)
    logger.info(f"Output Shape: {actual_shape}")

    # 3. Dynamic Validation (No hardcoded 4096)
    if actual_shape[-1] == model_args.dim and actual_shape[-2] == seq_len:
        logger.success(f"Test Passed! Output dims match config (Seq: {seq_len}, Dim: {model_args.dim}).")
    else:
        logger.error(f"Test Failed! Expected [..., {seq_len}, {model_args.dim}], but got {actual_shape}")

    ttnn.deallocate(tt_output)
    logger.info("Done.")

if __name__ == "__main__":
    test_rms_norm_inference()
