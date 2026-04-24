#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for Upsample1d module.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.modules.conditional_unet1d import (
    Upsample1d,
)
from navsim.agents.diffusiondrive.modules.conditional_unet1d_ttsim import (
    Upsample1d_TTSIM,
)

import ttsim.front.functional.op as F


def inject_weights(ttsim_block, pytorch_block):
    """
    Inject ConvTranspose1d weights [C_in, C_out, 4] into
    ConvTranspose2d weight [C_in, C_out, 4, 4].

    On (N,C,1,L) input with padding=(2,1), output_padding=(1,0), stride=(1,2):
    only kernel row kh=2 (= pad_h) touches the single data row.
    """
    w1d = pytorch_block.conv.weight.detach().cpu().numpy()  # [C_in, C_out, 4]
    K = w1d.shape[2]
    C_in, C_out = w1d.shape[0], w1d.shape[1]
    w2d = np.zeros((C_in, C_out, K, K), dtype=np.float32)
    # For transposed conv with pad_h=2, the active row is kh=pad_h=2
    w2d[:, :, 2, :] = w1d
    ttsim_block.conv_t.params[0][1].data = w2d

    # Bias
    if pytorch_block.conv.bias is not None:
        b = pytorch_block.conv.bias.detach().cpu().numpy().astype(np.float32)
        ttsim_block.bias.params[0][1].data = b.reshape(1, -1, 1, 1)


def main():
    print("=" * 70)
    print("Upsample1d Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    dim = 128
    batch_size = 2
    seq_len = 8  # will become 16 after upsample

    # --- PyTorch ---
    print("\n--- PyTorch Upsample1d ---")
    model_pt = Upsample1d(dim)
    model_pt.eval()

    x_data = np.random.randn(batch_size, dim, seq_len).astype(np.float32)
    x_pt = torch.from_numpy(x_data)

    with torch.no_grad():
        out_pt = model_pt(x_pt)
    out_pt_np = out_pt.numpy()

    print(f"Input shape:  {x_data.shape}")
    print(f"Output shape: {out_pt_np.shape}")
    print(
        f"Output stats: min={out_pt_np.min():.6f}, max={out_pt_np.max():.6f}, mean={out_pt_np.mean():.6f}"
    )

    # --- TTSIM ---
    print("\n--- TTSIM Upsample1d ---")
    model_ttsim = Upsample1d_TTSIM("us1d", dim)

    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    x_ttsim = F._from_data("input", x_data)
    x_ttsim.link_module = model_ttsim

    out_ttsim = model_ttsim(x_ttsim)

    print(f"Input shape:  {x_ttsim.shape}")
    print(f"Output shape: {out_ttsim.shape}")

    # --- Comparison ---
    if out_ttsim.data is not None:
        print(
            f"Output stats: min={out_ttsim.data.min():.6f}, max={out_ttsim.data.max():.6f}, mean={out_ttsim.data.mean():.6f}"
        )

        print("\n--- Numerical Comparison ---")
        atol, rtol = 1e-4, 1e-4
        print(f"Tolerance: atol={atol}, rtol={rtol}")

        diff = np.abs(out_pt_np - out_ttsim.data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Max absolute difference:  {max_diff:.10f}")
        print(f"  Mean absolute difference: {mean_diff:.10f}")

        shape_match = list(out_pt_np.shape) == list(out_ttsim.shape)
        print(
            f"\n  Shape match: {'[PASS]' if shape_match else '[FAIL]'}  PT={out_pt_np.shape}  TTSIM={out_ttsim.shape}"
        )

        is_close = np.allclose(out_pt_np, out_ttsim.data, atol=atol, rtol=rtol)

        print("\n" + "=" * 70)
        if is_close and shape_match:
            print("OVERALL: [PASS] PASS - TTSIM matches PyTorch")
        else:
            print("OVERALL: [FAIL] FAIL - Differences exceed tolerance")
        print("=" * 70)
    else:
        print("  [WARN] SKIPPED: No TTSIM data available")

    print()


if __name__ == "__main__":
    main()
