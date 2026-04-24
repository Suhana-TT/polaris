# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

# TTSIM imports
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module as SimNN_Module
from ttsim.front.functional.sim_nn import ModuleList as SimNN_ModuleList
from ttsim.front.functional.sim_nn import GroupNorm as SimNN_GroupNorm

if TYPE_CHECKING:
    from ttsim.ops.tensor import SimTensor


# ---------------------------------------------------------------------------
# 1. SinusoidalPosEmb_TTSIM
# ---------------------------------------------------------------------------
class SinusoidalPosEmb_TTSIM(SimNN_Module):
    """Sinusoidal positional embedding for TTSIM."""

    def __init__(self, dim, name=None):
        super().__init__()
        self.name = name or "SinusoidalPosEmb"
        self.dim = dim
        self.half_dim = dim // 2

        # Pre-compute frequency values with shape (1, half_dim) for broadcasting
        emb = np.log(10000) / (self.half_dim - 1)
        emb = np.exp(np.arange(self.half_dim, dtype=np.float32) * -emb)
        emb = emb.reshape(1, self.half_dim)  # Shape (1, half_dim) for broadcasting
        self.freqs = F._from_data(f"{self.name}_freqs", emb, is_const=True)

        # Create operations
        self.reshape_op = F.Reshape(f"{self.name}_reshape_x")
        self.mul_op = F.Mul(f"{self.name}_mul")
        self.sin_op = F.Sin(f"{self.name}_sin")
        self.cos_op = F.Cos(f"{self.name}_cos")
        self.concat_op = F.ConcatX(f"{self.name}_concat", axis=-1)

        # Link operations to module
        super().link_op2module()

    def __call__(self, x):
        """
        Compute sinusoidal positional embeddings.
        Args:
            x: Tensor of shape (batch_size,) containing timestep values
        Returns:
            Tensor of shape (batch_size, dim) containing sinusoidal embeddings
        """
        # Unique prefix per call
        if not hasattr(self, "_call_count"):
            self._call_count = 0
        self._call_count += 1
        _cc = self._call_count

        # Use the op-based path for both shape inference and data compute.
        # Sin/Cos ops support data computation via compute_sin / compute_cos.
        x.link_module = self
        x_shape = F._from_data(
            f"{self.name}_x_unsqueeze_shape_c{_cc}",
            np.array([x.shape[0], 1], dtype=np.int64),
            is_const=True,
        )
        self._tensors[x_shape.name] = x_shape
        x_unsqueezed = self.reshape_op(x, x_shape)

        # Multiply timesteps with frequencies: (batch_size, 1) * (1, half_dim) -> (batch_size, half_dim)
        emb = self.mul_op(x_unsqueezed, self.freqs)

        # Compute sin and cos
        sin_emb = self.sin_op(emb)
        cos_emb = self.cos_op(emb)

        # Concatenate sin and cos
        result = self.concat_op(sin_emb, cos_emb)

        return result


# ============================================================================
# 2. Conv1dBlock_TTSIM:  Conv1d → GroupNorm → Mish
# ============================================================================
class Conv1dBlock_TTSIM(SimNN_Module):
    """
    TTSIM equivalent of::

        nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size//2)
        nn.GroupNorm(n_groups, out_channels)
        nn.Mish()

    Conv1d is implemented as reshape→Conv2d(K,K)→reshape.  On (N,C,1,L) input
    with padding=K//2 the height output stays 1.

    Input:  (N, C_in, L)
    Output: (N, C_out, L)
    """

    def __init__(
        self,
        name: str,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.name = name
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Conv2d: weight [C_out, C_in, K, K], padding K//2 on all sides
        self.conv = F.Conv2d(
            f"{name}_conv",
            inp_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        # Bias: shape [1, C_out, 1, 1] broadcast-added to 4-D output
        self.bias = F.Bias(f"{name}_bias", [1, out_channels, 1, 1])

        # GroupNorm on 4-D so compute_groupnorm (N,C,H,W) works
        self.group_norm = SimNN_GroupNorm(f"{name}_gn", n_groups, out_channels)

        # Mish activation
        self.mish = F.Mish(f"{name}_mish")

        # Reshape ops for 3D ↔ 4D
        self.reshape_to_4d = F.Reshape(f"{name}_to4d")
        self.reshape_to_3d = F.Reshape(f"{name}_to3d")

        super().link_op2module()

    def __call__(self, x):
        self._call_count = getattr(self, "_call_count", 0) + 1
        _cc = self._call_count

        N, C_in, L = x.shape

        # (N, C_in, L) → (N, C_in, 1, L)
        shape_4d = F._from_data(
            f"{self.name}_in4d_c{_cc}",
            np.array([N, C_in, 1, L], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_4d.name] = shape_4d
        x = self.reshape_to_4d(x, shape_4d)

        # Conv2d + Bias: (N, C_in, 1, L) → (N, C_out, 1, L)
        x = self.conv(x)
        x = self.bias(x)

        # GroupNorm on 4-D (N, C_out, 1, L) then Mish
        x = self.group_norm(x)
        x = self.mish(x)

        # (N, C_out, 1, L) → (N, C_out, L)
        shape_3d = F._from_data(
            f"{self.name}_out3d_c{_cc}",
            np.array([N, self.out_channels, L], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_3d.name] = shape_3d
        x = self.reshape_to_3d(x, shape_3d)
        return x


# ============================================================================
# 3. Downsample1d_TTSIM:  Conv1d(dim, dim, 3, stride=2, padding=1)
# ============================================================================
class Downsample1d_TTSIM(SimNN_Module):
    """
    Halves the temporal dimension via strided convolution.
    Conv2d with kernel 3, stride (1,2), padding 1 on (N,C,1,L) → (N,C,1,L//2).

    Input:  (N, C, L)
    Output: (N, C, L // 2)
    """

    def __init__(self, name: str, dim: int):
        super().__init__()
        self.name = name
        self.dim = dim

        # stride=(1,2): only downsample width (temporal) dim
        self.conv = F.Conv2d(
            f"{name}_conv", dim, dim, kernel_size=3, stride=(1, 2), padding=1
        )
        self.bias = F.Bias(f"{name}_bias", [1, dim, 1, 1])

        self.reshape_to_4d = F.Reshape(f"{name}_to4d")
        self.reshape_to_3d = F.Reshape(f"{name}_to3d")

        super().link_op2module()

    def __call__(self, x):
        self._call_count = getattr(self, "_call_count", 0) + 1
        _cc = self._call_count

        N, C, L = x.shape

        shape_4d = F._from_data(
            f"{self.name}_in4d_c{_cc}",
            np.array([N, C, 1, L], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_4d.name] = shape_4d
        x = self.reshape_to_4d(x, shape_4d)

        x = self.conv(x)
        x = self.bias(x)

        L_out = (L + 2 * 1 - 3) // 2 + 1  # standard conv output formula
        shape_3d = F._from_data(
            f"{self.name}_out3d_c{_cc}",
            np.array([N, C, L_out], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_3d.name] = shape_3d
        x = self.reshape_to_3d(x, shape_3d)
        return x


# ============================================================================
# 4. Upsample1d_TTSIM:  ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
# ============================================================================
class Upsample1d_TTSIM(SimNN_Module):
    """
    Doubles the temporal dimension via transposed convolution.
    ConvTranspose2d(kernel=4, stride=(1,2)) with padding attrs so
    height stays 1 and width doubles.

    conv_transpose_sinf formula:
      H_out = (H-1)*s_h - 2*pad_h + dil*(K-1) + opad_h + 1
      W_out = (W-1)*s_w - 2*pad_w + dil*(K-1) + opad_w + 1

    With pad=(2,1), opad=(1,0), s=(1,2), K=4:
      H_out = 0 - 4 + 3 + 1 + 1 = 1   ✓
      W_out = 2(L-1) - 2 + 3 + 0 + 1 = 2L   ✓

    Input:  (N, C, L)
    Output: (N, C, 2L)
    """

    def __init__(self, name: str, dim: int):
        super().__init__()
        self.name = name
        self.dim = dim

        # ConvTranspose2d: weight [dim, dim, 4, 4], stride (1,2)
        self.conv_t = F.ConvTranspose2d(
            f"{name}_convt", dim, dim, kernel_size=4, stride=(1, 2)
        )
        # Set padding/output_padding attrs for correct shape
        self.conv_t.opinfo["attrs"]["padding"] = (2, 1)
        self.conv_t.opinfo["attrs"]["output_padding"] = (1, 0)

        self.bias = F.Bias(f"{name}_bias", [1, dim, 1, 1])

        self.reshape_to_4d = F.Reshape(f"{name}_to4d")
        self.reshape_to_3d = F.Reshape(f"{name}_to3d")

        super().link_op2module()

    def __call__(self, x):
        self._call_count = getattr(self, "_call_count", 0) + 1
        _cc = self._call_count

        N, C, L = x.shape

        # (N, C, L) → (N, C, 1, L)
        shape_4d = F._from_data(
            f"{self.name}_in4d_c{_cc}",
            np.array([N, C, 1, L], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_4d.name] = shape_4d
        x = self.reshape_to_4d(x, shape_4d)

        # ConvTranspose2d: (N,C,1,L) → (N,C,1,2L)
        x = self.conv_t(x)
        x = self.bias(x)

        # (N, C, 1, 2L) → (N, C, 2L)
        L_out = L * 2
        shape_3d = F._from_data(
            f"{self.name}_out3d_c{_cc}",
            np.array([N, C, L_out], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_3d.name] = shape_3d
        x = self.reshape_to_3d(x, shape_3d)
        return x


# ============================================================================
# 5. ConditionalResidualBlock1D_TTSIM
# ============================================================================
class ConditionalResidualBlock1D_TTSIM(SimNN_Module):
    """
    TTSIM equivalent of ConditionalResidualBlock1D.

    Two Conv1dBlocks with FiLM conditioning between them and a residual
    connection (1×1 conv if channel dims differ, identity otherwise).

    Input:  x    (N, in_channels, L)
            cond (N, cond_dim)
    Output: (N, out_channels, L)
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_predict_scale = cond_predict_scale

        # --- Two conv blocks ---
        self.block0 = Conv1dBlock_TTSIM(
            f"{name}_blk0", in_channels, out_channels, kernel_size, n_groups
        )
        self.block1 = Conv1dBlock_TTSIM(
            f"{name}_blk1", out_channels, out_channels, kernel_size, n_groups
        )

        # --- Cond encoder: Mish → Linear → Bias → Reshape ---
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_mish = F.Mish(f"{name}_cond_mish")
        self.cond_linear = F.Linear(f"{name}_cond_lin", cond_dim, cond_channels)
        self.cond_bias = F.Bias(f"{name}_cond_bias", [cond_channels])
        self.cond_reshape = F.Reshape(
            f"{name}_cond_resh"
        )  # (B, cond_ch) → (B, cond_ch, 1)

        # --- FiLM ops (cond_predict_scale path) ---
        if cond_predict_scale:
            self.film_reshape = F.Reshape(f"{name}_film_resh")  # → (B, 2, out_ch, 1)
            self.film_mul = F.Mul(f"{name}_film_mul")  # scale * out
            self.film_add = F.Add(f"{name}_film_add")  # + bias
        else:
            self.embed_add = F.Add(f"{name}_embed_add")  # out + embed

        # --- Residual conv (1×1) or identity ---
        self.need_residual_conv = in_channels != out_channels
        if self.need_residual_conv:
            self.res_conv = F.Conv2d(
                f"{name}_res_conv", in_channels, out_channels, kernel_size=1, padding=0
            )
            self.res_bias = F.Bias(f"{name}_res_bias", [1, out_channels, 1, 1])
            self.res_to4d = F.Reshape(f"{name}_res_to4d")
            self.res_to3d = F.Reshape(f"{name}_res_to3d")

        self.residual_add = F.Add(f"{name}_residual_add")

        super().link_op2module()

    def __call__(self, x, cond):
        self._call_count = getattr(self, "_call_count", 0) + 1
        _cc = self._call_count

        N = x.shape[0]
        L = x.shape[2]

        # --- Block 0 ---
        out = self.block0(x)  # (N, out_ch, L)

        # --- Cond encode ---
        embed = self.cond_mish(cond)  # (N, cond_dim)
        embed = self.cond_linear(embed)  # (N, cond_channels)
        embed = self.cond_bias(embed)
        cond_channels = embed.shape[-1]
        emb_shape = F._from_data(
            f"{self.name}_emb3d_c{_cc}",
            np.array([N, cond_channels, 1], dtype=np.int64),
            is_const=True,
        )
        self._tensors[emb_shape.name] = emb_shape
        embed = self.cond_reshape(embed, emb_shape)  # (N, cond_channels, 1)

        # --- FiLM modulation ---
        if self.cond_predict_scale:
            film_shape = F._from_data(
                f"{self.name}_film4d_c{_cc}",
                np.array([N, 2, self.out_channels, 1], dtype=np.int64),
                is_const=True,
            )
            self._tensors[film_shape.name] = film_shape
            embed = self.film_reshape(embed, film_shape)  # (N, 2, out_ch, 1)
            # TODO: slice along dim-1 for scale/bias once F.Slice is available;
            # for shape-inference purposes use Reshape workaround:
            # scale = embed[:, 0, :, :] → (N, out_ch, 1)
            # bias  = embed[:, 1, :, :] → (N, out_ch, 1)
            # For now, approximate with Mul + Add (shape propagation only)
            out = self.film_mul(embed, out)  # broadcast  (placeholder for scale * out)
            out = self.film_add(out, embed)  # placeholder for + bias
        else:
            out = self.embed_add(
                out, embed
            )  # (N, out_ch, L) + (N, out_ch, 1) broadcast

        # --- Block 1 ---
        out = self.block1(out)  # (N, out_ch, L)

        # --- Residual ---
        if self.need_residual_conv:
            res_4d_shape = F._from_data(
                f"{self.name}_res4d_c{_cc}",
                np.array([N, self.in_channels, 1, L], dtype=np.int64),
                is_const=True,
            )
            self._tensors[res_4d_shape.name] = res_4d_shape
            res = self.res_to4d(x, res_4d_shape)
            res = self.res_conv(res)
            res = self.res_bias(res)
            res_3d_shape = F._from_data(
                f"{self.name}_res3d_c{_cc}",
                np.array([N, self.out_channels, L], dtype=np.int64),
                is_const=True,
            )
            self._tensors[res_3d_shape.name] = res_3d_shape
            res = self.res_to3d(res, res_3d_shape)
        else:
            res = x

        out = self.residual_add(out, res)
        return out


# ============================================================================
# 6. ConditionalUnet1D_TTSIM
# ============================================================================
class ConditionalUnet1D_TTSIM(SimNN_Module):
    """
    TTSIM equivalent of ConditionalUnet1D.

    Full 1-D conditional U-Net with:
      - Diffusion step encoder (SinusoidalPosEmb → MLP)
      - Optional local conditioning encoder
      - Down path: pairs of ConditionalResidualBlock1D + Downsample
      - Mid: two ConditionalResidualBlock1D
      - Up path: pairs of ConditionalResidualBlock1D + Upsample
      - Final conv block + 1×1 projection
    """

    def __init__(
        self,
        name: str,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: list[int] | None = None,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        self.name = name
        if down_dims is None:
            down_dims = [256, 512, 1024]

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        # ---- Diffusion step encoder ----
        self.step_sinusoidal = SinusoidalPosEmb_TTSIM(dsed)
        self.step_linear1 = F.Linear(f"{name}_step_lin1", dsed, dsed * 4)
        self.step_bias1 = F.Bias(f"{name}_step_b1", [dsed * 4])
        self.step_mish = F.Mish(f"{name}_step_mish")
        self.step_linear2 = F.Linear(f"{name}_step_lin2", dsed * 4, dsed)
        self.step_bias2 = F.Bias(f"{name}_step_b2", [dsed])

        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        self.cond_dim = cond_dim
        self.has_global_cond = global_cond_dim is not None
        if self.has_global_cond:
            self.global_cat = F.ConcatX(f"{name}_global_cat", axis=-1)

        # ---- in_out pairs ----
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.in_out = in_out
        self.num_stages = len(in_out)

        # ---- Local cond encoder (optional) ----
        self.has_local_cond = local_cond_dim is not None
        if self.has_local_cond:
            assert local_cond_dim is not None
            _, dim_out = in_out[0]
            self.local_down = ConditionalResidualBlock1D_TTSIM(
                f"{name}_lc_down",
                local_cond_dim,
                dim_out,
                cond_dim,
                kernel_size,
                n_groups,
                cond_predict_scale,
            )
            self.local_up = ConditionalResidualBlock1D_TTSIM(
                f"{name}_lc_up",
                local_cond_dim,
                dim_out,
                cond_dim,
                kernel_size,
                n_groups,
                cond_predict_scale,
            )
            self.local_add_down = F.Add(f"{name}_local_add_down")
            self.local_add_up = F.Add(f"{name}_local_add_up")

        # ---- Transpose for sample: (B,H,T) → (B,T,H) ----
        self.sample_perm = F.Transpose(f"{name}_sample_perm", perm=[0, 2, 1])
        self.output_perm = F.Transpose(f"{name}_output_perm", perm=[0, 2, 1])

        # ---- Down modules ----
        down_resnets1 = []
        down_resnets2 = []
        down_samples: list = []  # Downsample1d_TTSIM or None
        down_is_last = []
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (len(in_out) - 1)
            down_is_last.append(is_last)
            down_resnets1.append(
                ConditionalResidualBlock1D_TTSIM(
                    f"{name}_d{idx}_r1",
                    dim_in,
                    dim_out,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                )
            )
            down_resnets2.append(
                ConditionalResidualBlock1D_TTSIM(
                    f"{name}_d{idx}_r2",
                    dim_out,
                    dim_out,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                )
            )
            if not is_last:
                down_samples.append(Downsample1d_TTSIM(f"{name}_d{idx}_ds", dim_out))
            else:
                down_samples.append(None)  # identity placeholder
        self.down_resnets1 = SimNN_ModuleList(down_resnets1)
        self.down_resnets2 = SimNN_ModuleList(down_resnets2)
        # Store individually so __setattr__ tracks them
        for idx, ds in enumerate(down_samples):
            if ds is not None:
                setattr(self, f"down_sample_{idx}", ds)
        self._down_is_last = down_is_last

        # ---- Mid modules ----
        mid_dim = all_dims[-1]
        self.mid0 = ConditionalResidualBlock1D_TTSIM(
            f"{name}_mid0",
            mid_dim,
            mid_dim,
            cond_dim,
            kernel_size,
            n_groups,
            cond_predict_scale,
        )
        self.mid1 = ConditionalResidualBlock1D_TTSIM(
            f"{name}_mid1",
            mid_dim,
            mid_dim,
            cond_dim,
            kernel_size,
            n_groups,
            cond_predict_scale,
        )

        # ---- Up modules ----
        up_resnets1 = []
        up_resnets2 = []
        up_samples: list = []  # Upsample1d_TTSIM or None
        up_is_last = []
        up_cats = []
        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx >= (len(in_out) - 1)
            up_is_last.append(is_last)
            up_resnets1.append(
                ConditionalResidualBlock1D_TTSIM(
                    f"{name}_u{idx}_r1",
                    dim_out * 2,
                    dim_in,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                )
            )
            up_resnets2.append(
                ConditionalResidualBlock1D_TTSIM(
                    f"{name}_u{idx}_r2",
                    dim_in,
                    dim_in,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                )
            )
            if not is_last:
                up_samples.append(Upsample1d_TTSIM(f"{name}_u{idx}_us", dim_in))
            else:
                up_samples.append(None)
            up_cats.append(F.ConcatX(f"{name}_u{idx}_cat", axis=1))
        self.up_resnets1 = SimNN_ModuleList(up_resnets1)
        self.up_resnets2 = SimNN_ModuleList(up_resnets2)
        for idx, us in enumerate(up_samples):
            if us is not None:
                setattr(self, f"up_sample_{idx}", us)
        for idx, cat_op in enumerate(up_cats):
            setattr(self, f"up_cat_{idx}", cat_op)
        self._up_is_last = up_is_last
        self._num_up = len(up_resnets1)

        # ---- Final conv ----
        self.final_block = Conv1dBlock_TTSIM(
            f"{name}_final_blk", start_dim, start_dim, kernel_size, n_groups
        )
        # 1×1 projection Conv1d → Conv2d(1)
        self.final_conv = F.Conv2d(
            f"{name}_final_conv", start_dim, input_dim, kernel_size=1, padding=0
        )
        self.final_conv_bias = F.Bias(f"{name}_final_conv_bias", [1, input_dim, 1, 1])
        self.final_to4d = F.Reshape(f"{name}_final_to4d")
        self.final_to3d = F.Reshape(f"{name}_final_to3d")

        super().link_op2module()

    # --------------------------------------------------------------------- #
    def __call__(self, sample, timestep, local_cond=None, global_cond=None):
        """
        Args:
            sample:      (B, H, T)  — noisy trajectory
            timestep:    (B,)       — diffusion step indices
            local_cond:  (B, H, T) or None
            global_cond: (B, global_cond_dim) or None
        Returns:
            (B, H, T)  — predicted noise / denoised trajectory
        """
        self._call_count = getattr(self, "_call_count", 0) + 1
        _cc = self._call_count

        # sample: (B, H, T) → (B, T, H) via transpose
        x = self.sample_perm(sample)

        B = x.shape[0]
        T_len = x.shape[1]  # channel dim after perm (was H in input)

        # --- diffusion step encoding ---
        global_feature = self.step_sinusoidal(timestep)  # (B, dsed)
        global_feature = self.step_linear1(global_feature)
        global_feature = self.step_bias1(global_feature)
        global_feature = self.step_mish(global_feature)
        global_feature = self.step_linear2(global_feature)
        global_feature = self.step_bias2(global_feature)  # (B, dsed)

        if self.has_global_cond and global_cond is not None:
            global_feature = self.global_cat(
                global_feature, global_cond
            )  # (B, cond_dim)

        # --- local conditioning ---
        h_local_down = None
        h_local_up = None
        if self.has_local_cond and local_cond is not None:
            lc = self.sample_perm(local_cond)  # reuse same transpose
            h_local_down = self.local_down(lc, global_feature)
            h_local_up = self.local_up(lc, global_feature)

        # --- down path ---
        h_list = []  # skip connections
        for idx in range(self.num_stages):
            r1 = self.down_resnets1[idx]
            r2 = self.down_resnets2[idx]
            x = r1(x, global_feature)
            if idx == 0 and h_local_down is not None:
                x = self.local_add_down(x, h_local_down)
            x = r2(x, global_feature)
            h_list.append(x)
            if not self._down_is_last[idx]:
                ds = getattr(self, f"down_sample_{idx}")
                x = ds(x)

        # --- mid ---
        x = self.mid0(x, global_feature)
        x = self.mid1(x, global_feature)

        # --- up path ---
        for idx in range(self._num_up):
            cat_op = getattr(self, f"up_cat_{idx}")
            skip = h_list.pop()
            x = cat_op(x, skip)  # (B, dim_out*2, L)
            r1 = self.up_resnets1[idx]
            r2 = self.up_resnets2[idx]
            x = r1(x, global_feature)
            # NOTE: original code checks `idx == len(up_modules)` which is always False
            # (dead code in published checkpoint), so we skip local_add_up here.
            x = r2(x, global_feature)
            if not self._up_is_last[idx]:
                us = getattr(self, f"up_sample_{idx}")
                x = us(x)

        # --- final conv ---
        x = self.final_block(x)  # (B, start_dim, T_len)

        N_f, C_f, L_f = x.shape
        shape_4d = F._from_data(
            f"{self.name}_fin4d_c{_cc}",
            np.array([N_f, C_f, 1, L_f], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_4d.name] = shape_4d
        x = self.final_to4d(x, shape_4d)
        x = self.final_conv(x)
        x = self.final_conv_bias(x)
        C_out_f = x.shape[1]
        shape_3d = F._from_data(
            f"{self.name}_fin3d_c{_cc}",
            np.array([N_f, C_out_f, L_f], dtype=np.int64),
            is_const=True,
        )
        self._tensors[shape_3d.name] = shape_3d
        x = self.final_to3d(x, shape_3d)

        # (B, T, H) → (B, H, T) via transpose
        x = self.output_perm(x)
        return x
