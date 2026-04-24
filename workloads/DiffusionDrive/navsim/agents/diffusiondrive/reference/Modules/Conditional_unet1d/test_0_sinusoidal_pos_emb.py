#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for SinusoidalPosEmb (conditional_unet1d copy).
Tests sinusoidal positional embedding against PyTorch reference.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.modules.conditional_unet1d_ttsim import (
    SinusoidalPosEmb_TTSIM,
)

import ttsim.front.functional.op as F


class SinusoidalPosEmb_PyTorch(nn.Module):
    """PyTorch reference implementation of Sinusoidal Positional Embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


def main():
    print("=" * 70)
    print("SinusoidalPosEmb Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    dim = 256
    batch_size = 4

    # --- PyTorch ---
    print("\n--- PyTorch SinusoidalPosEmb ---")
    model_pt = SinusoidalPosEmb_PyTorch(dim=dim)
    model_pt.eval()

    timesteps_data = np.random.randint(0, 1000, size=(batch_size,), dtype=np.int64)
    timesteps_pt = torch.from_numpy(timesteps_data)

    with torch.no_grad():
        output_pt = model_pt(timesteps_pt)
    output_pt_np = output_pt.numpy()

    print(f"Input (timesteps) shape: {timesteps_pt.shape}")
    print(f"Timesteps: {timesteps_data}")
    print(f"Output shape: {output_pt_np.shape}")
    print(
        f"Output stats: min={output_pt_np.min():.6f}, max={output_pt_np.max():.6f}, mean={output_pt_np.mean():.6f}"
    )

    # --- TTSIM ---
    print("\n--- TTSIM SinusoidalPosEmb ---")
    model_ttsim = SinusoidalPosEmb_TTSIM(dim=dim)

    timesteps_ttsim = F._from_data("timesteps", timesteps_data.astype(np.float32))
    timesteps_ttsim.link_module = model_ttsim

    output_ttsim = model_ttsim(timesteps_ttsim)

    print(f"Input (timesteps) shape: {timesteps_ttsim.shape}")
    print(f"Output shape: {output_ttsim.shape}")

    # --- Comparison ---
    if output_ttsim.data is not None:
        print(
            f"Output stats: min={output_ttsim.data.min():.6f}, max={output_ttsim.data.max():.6f}, mean={output_ttsim.data.mean():.6f}"
        )

        print("\n--- Numerical Comparison ---")
        atol, rtol = 1e-4, 1e-4
        print(f"Tolerance: atol={atol}, rtol={rtol}")

        diff = np.abs(output_pt_np - output_ttsim.data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"  Max absolute difference:  {max_diff:.10f}")
        print(f"  Mean absolute difference: {mean_diff:.10f}")

        shape_match = list(output_pt_np.shape) == list(output_ttsim.shape)
        print(
            f"\n  Shape match: {'[PASS]' if shape_match else '[FAIL]'}  PT={output_pt_np.shape}  TTSIM={output_ttsim.shape}"
        )

        is_close = np.allclose(output_pt_np, output_ttsim.data, atol=atol, rtol=rtol)

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
