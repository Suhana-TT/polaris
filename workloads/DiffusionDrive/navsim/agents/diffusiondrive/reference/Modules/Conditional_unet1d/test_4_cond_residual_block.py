#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for ConditionalResidualBlock1D module.
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
    ConditionalResidualBlock1D,
)
from navsim.agents.diffusiondrive.modules.conditional_unet1d_ttsim import (
    ConditionalResidualBlock1D_TTSIM,
)

import ttsim.front.functional.op as F

# ----- injection helpers -----


def inject_conv1d_block(ttsim_blk, pt_blk):
    """Inject Conv1dBlock weights (Conv1d + GroupNorm)."""
    # Conv1d → Conv2d
    pt_conv = pt_blk.block[0]
    w1d = pt_conv.weight.detach().cpu().numpy()
    K = w1d.shape[2]
    C_out, C_in = w1d.shape[:2]
    w2d = np.zeros((C_out, C_in, K, K), dtype=np.float32)
    w2d[:, :, K // 2, :] = w1d
    ttsim_blk.conv.params[0][1].data = w2d

    if pt_conv.bias is not None:
        ttsim_blk.bias.params[0][1].data = (
            pt_conv.bias.detach().cpu().numpy().reshape(1, -1, 1, 1).astype(np.float32)
        )

    # GroupNorm
    pt_gn = pt_blk.block[1]
    ttsim_blk.group_norm.weight.data = (
        pt_gn.weight.detach().cpu().numpy().astype(np.float32)
    )
    ttsim_blk.group_norm.bias.data = (
        pt_gn.bias.detach().cpu().numpy().astype(np.float32)
    )


def inject_residual_conv1d(ttsim_res_conv, ttsim_res_bias, pt_conv1d):
    """Inject residual 1×1 Conv1d into Conv2d(1)."""
    w1d = pt_conv1d.weight.detach().cpu().numpy()  # [C_out, C_in, 1]
    C_out, C_in, _ = w1d.shape
    w2d = np.zeros((C_out, C_in, 1, 1), dtype=np.float32)
    w2d[:, :, 0, :] = w1d  # K=1, K//2=0
    ttsim_res_conv.params[0][1].data = w2d

    if pt_conv1d.bias is not None:
        ttsim_res_bias.params[0][1].data = (
            pt_conv1d.bias.detach()
            .cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )


def inject_weights(ttsim_model, pt_model):
    """Inject all ConditionalResidualBlock1D weights."""
    # Two Conv1d blocks
    inject_conv1d_block(ttsim_model.block0, pt_model.blocks[0])
    inject_conv1d_block(ttsim_model.block1, pt_model.blocks[1])

    # Cond encoder: Sequential(Mish(), Linear(cond_dim, cond_channels), Rearrange)
    pt_linear = pt_model.cond_encoder[1]  # nn.Linear
    ttsim_model.cond_linear.params[0][1].data = (
        pt_linear.weight.detach().cpu().numpy().T.astype(np.float32)
    )
    ttsim_model.cond_bias.params[0][1].data = (
        pt_linear.bias.detach().cpu().numpy().astype(np.float32)
    )

    # Residual conv (1×1) — only when in != out channels
    if ttsim_model.need_residual_conv:
        inject_residual_conv1d(
            ttsim_model.res_conv, ttsim_model.res_bias, pt_model.residual_conv
        )


def main():
    print("=" * 70)
    print("ConditionalResidualBlock1D Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration (in_channels != out_channels to test residual conv)
    in_channels = 64
    out_channels = 128
    cond_dim = 256
    kernel_size = 3
    n_groups = 8
    cond_predict_scale = False
    batch_size = 2
    seq_len = 16

    # --- PyTorch ---
    print("\n--- PyTorch ConditionalResidualBlock1D ---")
    model_pt = ConditionalResidualBlock1D(
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=cond_predict_scale,
    )
    model_pt.eval()

    x_data = np.random.randn(batch_size, in_channels, seq_len).astype(np.float32)
    cond_data = np.random.randn(batch_size, cond_dim).astype(np.float32)

    with torch.no_grad():
        out_pt = model_pt(torch.from_numpy(x_data), torch.from_numpy(cond_data))
    out_pt_np = out_pt.numpy()

    print(f"Input x shape:    {x_data.shape}")
    print(f"Input cond shape: {cond_data.shape}")
    print(f"Output shape:     {out_pt_np.shape}")
    print(
        f"Output stats: min={out_pt_np.min():.6f}, max={out_pt_np.max():.6f}, mean={out_pt_np.mean():.6f}"
    )

    # --- TTSIM ---
    print("\n--- TTSIM ConditionalResidualBlock1D ---")
    model_ttsim = ConditionalResidualBlock1D_TTSIM(
        "cond_res",
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=cond_predict_scale,
    )

    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    x_ttsim = F._from_data("x_input", x_data)
    cond_ttsim = F._from_data("cond_input", cond_data)
    x_ttsim.link_module = model_ttsim
    cond_ttsim.link_module = model_ttsim

    out_ttsim = model_ttsim(x_ttsim, cond_ttsim)

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

    # === Test 2: Same channels (identity residual) ===
    print("\n\n" + "=" * 70)
    print("Test 2: in_channels == out_channels (identity residual)")
    print("=" * 70)

    torch.manual_seed(99)
    in_ch2 = 128
    out_ch2 = 128

    model_pt2 = ConditionalResidualBlock1D(
        in_ch2,
        out_ch2,
        cond_dim,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=cond_predict_scale,
    )
    model_pt2.eval()

    x2_data = np.random.randn(batch_size, in_ch2, seq_len).astype(np.float32)

    with torch.no_grad():
        out_pt2 = model_pt2(torch.from_numpy(x2_data), torch.from_numpy(cond_data))
    out_pt2_np = out_pt2.numpy()

    model_ttsim2 = ConditionalResidualBlock1D_TTSIM(
        "cond_res2",
        in_ch2,
        out_ch2,
        cond_dim,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=cond_predict_scale,
    )

    inject_weights(model_ttsim2, model_pt2)

    x2_ttsim = F._from_data("x2_input", x2_data)
    cond2_ttsim = F._from_data("cond2_input", cond_data)
    x2_ttsim.link_module = model_ttsim2
    cond2_ttsim.link_module = model_ttsim2

    out_ttsim2 = model_ttsim2(x2_ttsim, cond2_ttsim)

    if out_ttsim2.data is not None:
        diff2 = np.abs(out_pt2_np - out_ttsim2.data)
        max_diff2 = np.max(diff2)
        shape_match2 = list(out_pt2_np.shape) == list(out_ttsim2.shape)
        is_close2 = np.allclose(out_pt2_np, out_ttsim2.data, atol=atol, rtol=rtol)
        print(
            f"  Shape: PT={out_pt2_np.shape}  TTSIM={out_ttsim2.shape}  match={'[PASS]' if shape_match2 else '[FAIL]'}"
        )
        print(f"  Max diff: {max_diff2:.10f}")
        print("\n" + "=" * 70)
        if is_close2 and shape_match2:
            print("TEST 2: [PASS] PASS")
        else:
            print("TEST 2: [FAIL] FAIL")
        print("=" * 70)
    else:
        print("  [WARN] SKIPPED: No TTSIM data")

    print()


if __name__ == "__main__":
    main()
