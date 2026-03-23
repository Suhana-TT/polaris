#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
NumPy reference implementations for UniAD operations.
Used for validating ttsim workload implementations.
"""

import numpy as np

# ─── Basic ops ────────────────────────────────────────────────────────────────


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ─── Linear / matmul ──────────────────────────────────────────────────────────


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """x @ w.T + b.  w shape: [out_features, in_features]."""
    out = x @ w.T
    if b is not None:
        out = out + b
    return out


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


# ─── Normalisation ────────────────────────────────────────────────────────────


def batch_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray | None = None,
    var: np.ndarray | None = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """x: [N, C, H, W]."""
    if mean is None:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
    if var is None:
        var = x.var(axis=(0, 2, 3), keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    # gamma/beta: [C] -> broadcast over [N, C, H, W]
    return gamma[None, :, None, None] * x_norm + beta[None, :, None, None]


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Normalise over last dimension(s) matching gamma/beta shape."""
    ndim = gamma.ndim
    axes = tuple(range(x.ndim - ndim, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# ─── Convolution ──────────────────────────────────────────────────────────────


def conv2d(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> np.ndarray:
    """
    x: [N, C_in, H, W]
    w: [C_out, C_in/groups, kH, kW]
    returns: [N, C_out, H_out, W_out]
    """
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, kH, kW = w.shape
    assert C_in == C_in_per_group * groups

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_out = (x.shape[2] - dilation * (kH - 1) - 1) // stride + 1
    W_out = (x.shape[3] - dilation * (kW - 1) - 1) // stride + 1
    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    group_in = C_in // groups
    group_out = C_out // groups

    for g in range(groups):
        x_g = x[:, g * group_in : (g + 1) * group_in]
        w_g = w[g * group_out : (g + 1) * group_out]
        for n in range(N):
            for oc in range(group_out):
                for oh in range(H_out):
                    for ow in range(W_out):
                        h_start = oh * stride
                        w_start = ow * stride
                        patch = x_g[
                            n,
                            :,
                            h_start : h_start + kH * dilation : dilation,
                            w_start : w_start + kW * dilation : dilation,
                        ]
                        out[n, g * group_out + oc, oh, ow] = (patch * w_g[oc]).sum()

    if b is not None:
        out += b[None, :, None, None]
    return out


def max_pool2d(
    x: np.ndarray, kernel_size: int = 2, stride: int = 2, padding: int = 0
) -> np.ndarray:
    """x: [N, C, H, W]."""
    N, C, H, W = x.shape
    if padding > 0:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            constant_values=-np.inf,
        )
    H_out = (x.shape[2] - kernel_size) // stride + 1
    W_out = (x.shape[3] - kernel_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for oh in range(H_out):
        for ow in range(W_out):
            patch = x[
                :,
                :,
                oh * stride : oh * stride + kernel_size,
                ow * stride : ow * stride + kernel_size,
            ]
            out[:, :, oh, ow] = patch.max(axis=(2, 3))
    return out


def avg_pool2d(
    x: np.ndarray, kernel_size: int = 2, stride: int = 2, padding: int = 0
) -> np.ndarray:
    N, C, H, W = x.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = (x.shape[2] - kernel_size) // stride + 1
    W_out = (x.shape[3] - kernel_size) // stride + 1
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    for oh in range(H_out):
        for ow in range(W_out):
            patch = x[
                :,
                :,
                oh * stride : oh * stride + kernel_size,
                ow * stride : ow * stride + kernel_size,
            ]
            out[:, :, oh, ow] = patch.mean(axis=(2, 3))
    return out


# ─── Attention helpers ────────────────────────────────────────────────────────


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    q: [..., seq_q, head_dim]
    k: [..., seq_k, head_dim]
    v: [..., seq_k, head_dim]
    """
    d_k = q.shape[-1]
    scores = q @ k.swapaxes(-1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v


def multi_head_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    b_q: np.ndarray | None = None,
    b_k: np.ndarray | None = None,
    b_v: np.ndarray | None = None,
    b_o: np.ndarray | None = None,
    num_heads: int = 8,
) -> np.ndarray:
    """
    query: [B, S, E]
    Returns [B, S, E]
    """
    B, S, E = query.shape
    head_dim = E // num_heads

    Q = linear(query, w_q, b_q)  # [B, S, E]
    K = linear(key, w_k, b_k)
    V = linear(value, w_v, b_v)

    # split heads
    Q = Q.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)  # [B, nH, S, dH]
    K = K.reshape(B, -1, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, -1, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn_out = scaled_dot_product_attention(Q, K, V)  # [B, nH, S, dH]
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, E)
    return linear(attn_out, w_o, b_o)


def ffn(
    x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """Two-layer FFN with ReLU."""
    return linear(relu(linear(x, w1, b1)), w2, b2)


# ─── FPN upsample (nearest) ──────────────────────────────────────────────────


def upsample_nearest2x(x: np.ndarray) -> np.ndarray:
    """x: [N, C, H, W] -> [N, C, 2H, 2W]."""
    return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)


# ─── ResNet bottleneck ────────────────────────────────────────────────────────


def bottleneck(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,  # 1x1 conv
    w2: np.ndarray,
    b2: np.ndarray,  # 3x3 conv
    w3: np.ndarray,
    b3: np.ndarray,  # 1x1 conv
    gamma1: np.ndarray,
    beta1: np.ndarray,
    gamma2: np.ndarray,
    beta2: np.ndarray,
    gamma3: np.ndarray,
    beta3: np.ndarray,
    stride: int = 1,
    w_ds: np.ndarray | None = None,
    b_ds: np.ndarray | None = None,
    gamma_ds: np.ndarray | None = None,
    beta_ds: np.ndarray | None = None,
) -> np.ndarray:
    residual = x
    y = relu(batch_norm(conv2d(x, w1, None, stride=1, padding=0), gamma1, beta1))
    y = relu(batch_norm(conv2d(y, w2, None, stride=stride, padding=1), gamma2, beta2))
    y = batch_norm(conv2d(y, w3, None, stride=1, padding=0), gamma3, beta3)
    if w_ds is not None:
        residual = batch_norm(conv2d(x, w_ds, None, stride=stride, padding=0), gamma_ds, beta_ds)  # type: ignore[arg-type]
    return relu(y + residual)
