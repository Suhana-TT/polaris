#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for ConditionalUnet1D (full U-Net).
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
    ConditionalUnet1D,
)
from navsim.agents.diffusiondrive.modules.conditional_unet1d_ttsim import (
    ConditionalUnet1D_TTSIM,
)

import ttsim.front.functional.op as F

# ================================================================
# Weight injection helpers
# ================================================================


def _inject_conv1d_as_conv2d(ttsim_conv2d, ttsim_bias, pt_conv1d):
    """Conv1d weight [Co,Ci,K] → Conv2d weight [Co,Ci,K,K] centre row."""
    w1d = pt_conv1d.weight.detach().cpu().numpy()
    K = w1d.shape[2]
    Co, Ci = w1d.shape[:2]
    w2d = np.zeros((Co, Ci, K, K), dtype=np.float32)
    w2d[:, :, K // 2, :] = w1d
    ttsim_conv2d.params[0][1].data = w2d
    if pt_conv1d.bias is not None and ttsim_bias is not None:
        ttsim_bias.params[0][1].data = (
            pt_conv1d.bias.detach()
            .cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )


def _inject_conv1d_block(ttsim_blk, pt_blk):
    """Inject Conv1dBlock: Conv1d + GroupNorm."""
    _inject_conv1d_as_conv2d(ttsim_blk.conv, ttsim_blk.bias, pt_blk.block[0])
    pt_gn = pt_blk.block[1]
    ttsim_blk.group_norm.weight.data = (
        pt_gn.weight.detach().cpu().numpy().astype(np.float32)
    )
    ttsim_blk.group_norm.bias.data = (
        pt_gn.bias.detach().cpu().numpy().astype(np.float32)
    )


def _inject_cond_residual_block(ttsim_crb, pt_crb):
    """Inject ConditionalResidualBlock1D weights."""
    # Two conv blocks
    _inject_conv1d_block(ttsim_crb.block0, pt_crb.blocks[0])
    _inject_conv1d_block(ttsim_crb.block1, pt_crb.blocks[1])

    # Cond encoder: Sequential(Mish(), Linear, Rearrange)
    pt_lin = pt_crb.cond_encoder[1]
    ttsim_crb.cond_linear.params[0][1].data = (
        pt_lin.weight.detach().cpu().numpy().T.astype(np.float32)
    )
    ttsim_crb.cond_bias.params[0][1].data = (
        pt_lin.bias.detach().cpu().numpy().astype(np.float32)
    )

    # Residual conv 1×1
    if ttsim_crb.need_residual_conv:
        w1d = pt_crb.residual_conv.weight.detach().cpu().numpy()
        Co, Ci, _ = w1d.shape
        w2d = np.zeros((Co, Ci, 1, 1), dtype=np.float32)
        w2d[:, :, 0, :] = w1d
        ttsim_crb.res_conv.params[0][1].data = w2d
        if pt_crb.residual_conv.bias is not None:
            ttsim_crb.res_bias.params[0][1].data = (
                pt_crb.residual_conv.bias.detach()
                .cpu()
                .numpy()
                .reshape(1, -1, 1, 1)
                .astype(np.float32)
            )


def _inject_downsample(ttsim_ds, pt_ds):
    """Inject Downsample1d: Conv1d(dim,dim,3,2,1) → Conv2d."""
    w1d = pt_ds.conv.weight.detach().cpu().numpy()
    K = w1d.shape[2]
    C = w1d.shape[0]
    w2d = np.zeros((C, C, K, K), dtype=np.float32)
    w2d[:, :, K // 2, :] = w1d
    ttsim_ds.conv.params[0][1].data = w2d
    if pt_ds.conv.bias is not None:
        ttsim_ds.bias.params[0][1].data = (
            pt_ds.conv.bias.detach()
            .cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )


def _inject_upsample(ttsim_us, pt_us):
    """Inject Upsample1d: ConvTranspose1d(dim,dim,4,2,1) → ConvTranspose2d."""
    w1d = pt_us.conv.weight.detach().cpu().numpy()  # [C_in, C_out, 4]
    K = w1d.shape[2]
    Ci, Co = w1d.shape[:2]
    w2d = np.zeros((Ci, Co, K, K), dtype=np.float32)
    w2d[:, :, 2, :] = w1d  # active row = pad_h = 2
    ttsim_us.conv_t.params[0][1].data = w2d
    if pt_us.conv.bias is not None:
        ttsim_us.bias.params[0][1].data = (
            pt_us.conv.bias.detach()
            .cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )


def inject_unet_weights(ttsim_unet, pt_unet):
    """Inject all ConditionalUnet1D weights from PyTorch to TTSIM."""

    # --- Diffusion step encoder: Sequential(SinusoidalPosEmb, Linear, Mish, Linear) ---
    # SinusoidalPosEmb has no trainable weights
    pt_step_enc = pt_unet.diffusion_step_encoder
    ttsim_unet.step_linear1.params[0][1].data = (
        pt_step_enc[1].weight.detach().cpu().numpy().T.astype(np.float32)
    )
    ttsim_unet.step_bias1.params[0][1].data = (
        pt_step_enc[1].bias.detach().cpu().numpy().astype(np.float32)
    )
    ttsim_unet.step_linear2.params[0][1].data = (
        pt_step_enc[3].weight.detach().cpu().numpy().T.astype(np.float32)
    )
    ttsim_unet.step_bias2.params[0][1].data = (
        pt_step_enc[3].bias.detach().cpu().numpy().astype(np.float32)
    )

    # --- Local cond encoder (if present) ---
    if pt_unet.local_cond_encoder is not None and ttsim_unet.has_local_cond:
        _inject_cond_residual_block(
            ttsim_unet.local_down, pt_unet.local_cond_encoder[0]
        )
        _inject_cond_residual_block(ttsim_unet.local_up, pt_unet.local_cond_encoder[1])

    # --- Down modules ---
    for idx, pt_stage in enumerate(pt_unet.down_modules):
        pt_r1, pt_r2, pt_ds = pt_stage
        ttsim_r1 = ttsim_unet.down_resnets1[idx]
        ttsim_r2 = ttsim_unet.down_resnets2[idx]
        _inject_cond_residual_block(ttsim_r1, pt_r1)
        _inject_cond_residual_block(ttsim_r2, pt_r2)
        if not ttsim_unet._down_is_last[idx]:
            ttsim_ds = getattr(ttsim_unet, f"down_sample_{idx}")
            _inject_downsample(ttsim_ds, pt_ds)

    # --- Mid modules ---
    _inject_cond_residual_block(ttsim_unet.mid0, pt_unet.mid_modules[0])
    _inject_cond_residual_block(ttsim_unet.mid1, pt_unet.mid_modules[1])

    # --- Up modules ---
    for idx, pt_stage in enumerate(pt_unet.up_modules):
        pt_r1, pt_r2, pt_us = pt_stage
        ttsim_r1 = ttsim_unet.up_resnets1[idx]
        ttsim_r2 = ttsim_unet.up_resnets2[idx]
        _inject_cond_residual_block(ttsim_r1, pt_r1)
        _inject_cond_residual_block(ttsim_r2, pt_r2)
        if not ttsim_unet._up_is_last[idx]:
            ttsim_us = getattr(ttsim_unet, f"up_sample_{idx}")
            _inject_upsample(ttsim_us, pt_us)

    # --- Final conv: Conv1dBlock + Conv1d(start_dim, input_dim, 1) ---
    _inject_conv1d_block(ttsim_unet.final_block, pt_unet.final_conv[0])

    # Final 1×1 Conv1d → Conv2d
    pt_final_1x1 = pt_unet.final_conv[1]
    w1d = pt_final_1x1.weight.detach().cpu().numpy()  # [input_dim, start_dim, 1]
    Co, Ci, _ = w1d.shape
    w2d = np.zeros((Co, Ci, 1, 1), dtype=np.float32)
    w2d[:, :, 0, :] = w1d
    ttsim_unet.final_conv.params[0][1].data = w2d
    if pt_final_1x1.bias is not None:
        ttsim_unet.final_conv_bias.params[0][1].data = (
            pt_final_1x1.bias.detach()
            .cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )


# ================================================================
# Main
# ================================================================
def main():
    print("=" * 70)
    print("ConditionalUnet1D Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Use small dims to keep test fast
    input_dim = 4  # trajectory: (x, y, heading, ...)
    down_dims = [32, 64]
    dsed = 32  # diffusion step embed dim
    global_cond_dim = 16
    kernel_size = 3
    n_groups = 8
    cond_predict_scale = False
    batch_size = 2
    horizon = 8  # temporal length

    # --- PyTorch ---
    print("\n--- PyTorch ConditionalUnet1D ---")
    model_pt = ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=dsed,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=cond_predict_scale,
    )
    model_pt.eval()

    # Inputs – forward() expects sample as (B, T, input_dim) and internally
    # does einops.rearrange(sample, 'b h t -> b t h') → (B, input_dim, T).
    sample_data = np.random.randn(batch_size, horizon, input_dim).astype(
        np.float32
    )  # (B, T, input_dim)
    timestep_data = np.array([5, 10], dtype=np.int64)  # (B,)
    global_cond_data = np.random.randn(batch_size, global_cond_dim).astype(
        np.float32
    )  # (B, gcond)

    with torch.no_grad():
        out_pt = model_pt(
            sample=torch.from_numpy(sample_data),
            timestep=torch.from_numpy(timestep_data),
            global_cond=torch.from_numpy(global_cond_data),
        )
    out_pt_np = out_pt.numpy()

    print(f"Input sample shape:  {sample_data.shape}")
    print(f"Timestep:            {timestep_data}")
    print(f"Global cond shape:   {global_cond_data.shape}")
    print(f"Output shape:        {out_pt_np.shape}")
    print(
        f"Output stats: min={out_pt_np.min():.6f}, max={out_pt_np.max():.6f}, mean={out_pt_np.mean():.6f}"
    )

    # --- TTSIM ---
    print("\n--- TTSIM ConditionalUnet1D ---")
    model_ttsim = ConditionalUnet1D_TTSIM(
        "unet1d",
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=dsed,
        down_dims=down_dims,
        kernel_size=kernel_size,
        n_groups=n_groups,
        cond_predict_scale=cond_predict_scale,
    )

    print("Injecting weights...")
    inject_unet_weights(model_ttsim, model_pt)
    print("[PASS] Weight injection complete")

    sample_ttsim = F._from_data("sample", sample_data)
    timestep_ttsim = F._from_data("timestep", timestep_data.astype(np.float32))
    global_cond_ttsim = F._from_data("global_cond", global_cond_data)
    sample_ttsim.link_module = model_ttsim
    timestep_ttsim.link_module = model_ttsim
    global_cond_ttsim.link_module = model_ttsim

    out_ttsim = model_ttsim(sample_ttsim, timestep_ttsim, global_cond=global_cond_ttsim)

    print(f"Output shape: {out_ttsim.shape}")

    # --- Comparison ---
    if out_ttsim.data is not None:
        print(
            f"Output stats: min={out_ttsim.data.min():.6f}, max={out_ttsim.data.max():.6f}, mean={out_ttsim.data.mean():.6f}"
        )

        print("\n--- Numerical Comparison ---")
        atol, rtol = 1e-3, 1e-3  # slightly relaxed for deep network
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
            if not shape_match:
                print(
                    f"  Shape mismatch: PT={out_pt_np.shape} vs TTSIM={list(out_ttsim.shape)}"
                )
            if max_diff > atol:
                print(f"  Max diff {max_diff:.6e} > atol {atol}")
        print("=" * 70)
    else:
        print("  [WARN] SKIPPED: No TTSIM data available (shape-inference only)")
        # Still validate shapes
        shape_match = list(out_pt_np.shape) == list(out_ttsim.shape)
        print(
            f"  Shape match: {'[PASS]' if shape_match else '[FAIL]'}  PT={out_pt_np.shape}  TTSIM={list(out_ttsim.shape)}"
        )

    print()


if __name__ == "__main__":
    main()
