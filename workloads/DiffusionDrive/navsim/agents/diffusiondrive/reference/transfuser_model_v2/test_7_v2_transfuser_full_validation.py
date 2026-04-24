#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Full end-to-end validation for V2TransfuserModel: shape + numerical equivalence.

PyTorch vs TTSIM comparison for all non-trajectory outputs:
  - bev_semantic_map
  - agent_states
  - agent_labels

Trajectory is skipped here (TTSIM uses a mock scheduler); see test_6_trajectory_head.py.


"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.transfuser_model_v2 import (
    V2TransfuserModel as V2TransfuserModel_PyTorch,
)
from navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (
    V2TransfuserModel as V2TransfuserModel_TTSIM,
)

import ttsim.front.functional.op as F

# ---------------------------------------------------------------------------
# Weight injection helpers
# ---------------------------------------------------------------------------


def _inject_linear(ttsim_linear, ttsim_bias, pt_linear):
    ttsim_linear.params[0][1].data = (
        pt_linear.weight.detach().cpu().numpy().T.astype(np.float32)
    )
    ttsim_bias.params[0][1].data = (
        pt_linear.bias.detach().cpu().numpy().astype(np.float32)
    )


def _inject_layernorm(ttsim_ln, pt_ln):
    if pt_ln.weight is not None:
        ttsim_ln.params[0][1].data = (
            pt_ln.weight.detach().cpu().numpy().astype(np.float32)
        )
    if pt_ln.bias is not None:
        ttsim_ln.params[1][1].data = (
            pt_ln.bias.detach().cpu().numpy().astype(np.float32)
        )


def _inject_conv2d_weight_only(ttsim_conv, pt_conv):
    """Inject Conv2d WEIGHT only — for convs with a paired external F.Bias op."""
    ttsim_conv.params[0][1].data = (
        pt_conv.weight.detach().cpu().numpy().astype(np.float32)
    )


def _inject_conv2d_full(ttsim_conv, pt_conv):
    """Inject Conv2d weight + bias — for convs WITHOUT a separate F.Bias op."""
    ttsim_conv.params[0][1].data = (
        pt_conv.weight.detach().cpu().numpy().astype(np.float32)
    )
    if pt_conv.bias is not None and len(ttsim_conv.params) > 1:
        ttsim_conv.params[1][1].data = (
            pt_conv.bias.detach().cpu().numpy().astype(np.float32)
        )


def _inject_conv2d_bias_op(ttsim_bias_op, pt_conv, reshape=None):
    """Inject PyTorch conv bias into a standalone F.Bias op."""
    bias = pt_conv.bias.detach().cpu().numpy().astype(np.float32)
    ttsim_bias_op.params[0][1].data = bias.reshape(reshape) if reshape else bias


def _inject_bn2d(pt_bn, ttsim_bn):
    ttsim_bn.params[0][1].data = pt_bn.weight.data.cpu().numpy()
    ttsim_bn.params[1][1].data = pt_bn.bias.data.cpu().numpy()
    ttsim_bn.params[2][1].data = pt_bn.running_mean.cpu().numpy()
    ttsim_bn.params[3][1].data = pt_bn.running_var.cpu().numpy()


def _inject_embedding(ttsim_emb, pt_emb):
    ttsim_emb.params[0][1].data = (
        pt_emb.weight.detach().cpu().numpy().astype(np.float32)
    )


def _inject_mha(
    ttsim_q,
    ttsim_qb,
    ttsim_k,
    ttsim_kb,
    ttsim_v,
    ttsim_vb,
    ttsim_out,
    ttsim_outb,
    pt_mha,
    d_model,
):
    w = pt_mha.in_proj_weight.detach().cpu().numpy()
    b = pt_mha.in_proj_bias.detach().cpu().numpy()
    ttsim_q.params[0][1].data = w[:d_model, :].T.astype(np.float32)
    ttsim_qb.params[0][1].data = b[:d_model].astype(np.float32)
    ttsim_k.params[0][1].data = w[d_model : 2 * d_model, :].T.astype(np.float32)
    ttsim_kb.params[0][1].data = b[d_model : 2 * d_model].astype(np.float32)
    ttsim_v.params[0][1].data = w[2 * d_model :, :].T.astype(np.float32)
    ttsim_vb.params[0][1].data = b[2 * d_model :].astype(np.float32)
    ttsim_out.params[0][1].data = (
        pt_mha.out_proj.weight.detach().cpu().numpy().T.astype(np.float32)
    )
    ttsim_outb.params[0][1].data = (
        pt_mha.out_proj.bias.detach().cpu().numpy().astype(np.float32)
    )


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


def _inject_gpt(pt_gpt, ts_gpt):
    if hasattr(pt_gpt, "pos_emb"):
        ts_gpt.pos_emb.data = pt_gpt.pos_emb.data.cpu().numpy()
    for i, (pt_blk, ts_blk) in enumerate(zip(pt_gpt.blocks, ts_gpt.blocks)):
        ts_blk.ln1.params[0][1].data = pt_blk.ln1.weight.data.cpu().numpy()
        ts_blk.ln1.params[1][1].data = pt_blk.ln1.bias.data.cpu().numpy()
        ts_blk.attn.query.params[0][1].data = (
            pt_blk.attn.query.weight.data.cpu().numpy().T
        )
        ts_blk.attn.query_bias.params[0][
            1
        ].data = pt_blk.attn.query.bias.data.cpu().numpy()
        ts_blk.attn.key.params[0][1].data = pt_blk.attn.key.weight.data.cpu().numpy().T
        ts_blk.attn.key_bias.params[0][1].data = pt_blk.attn.key.bias.data.cpu().numpy()
        ts_blk.attn.value.params[0][1].data = (
            pt_blk.attn.value.weight.data.cpu().numpy().T
        )
        ts_blk.attn.value_bias.params[0][
            1
        ].data = pt_blk.attn.value.bias.data.cpu().numpy()
        ts_blk.attn.proj.params[0][1].data = (
            pt_blk.attn.proj.weight.data.cpu().numpy().T
        )
        ts_blk.attn.proj_bias.params[0][
            1
        ].data = pt_blk.attn.proj.bias.data.cpu().numpy()
        ts_blk.ln2.params[0][1].data = pt_blk.ln2.weight.data.cpu().numpy()
        ts_blk.ln2.params[1][1].data = pt_blk.ln2.bias.data.cpu().numpy()
        ts_blk.mlp_fc1.params[0][1].data = pt_blk.mlp[0].weight.data.cpu().numpy().T
        ts_blk.mlp_fc1_bias.params[0][1].data = pt_blk.mlp[0].bias.data.cpu().numpy()
        ts_blk.mlp_fc2.params[0][1].data = pt_blk.mlp[2].weight.data.cpu().numpy().T
        ts_blk.mlp_fc2_bias.params[0][1].data = pt_blk.mlp[2].bias.data.cpu().numpy()
    ts_gpt.ln_f.params[0][1].data = pt_gpt.ln_f.weight.data.cpu().numpy()
    ts_gpt.ln_f.params[1][1].data = pt_gpt.ln_f.bias.data.cpu().numpy()


def _inject_basic_block(pt_blk, ts_blk):
    _inject_conv2d_full(ts_blk.conv1, pt_blk.conv1)  # bias=False in PyTorch ResNet
    _inject_bn2d(pt_blk.bn1, ts_blk.bn1)
    _inject_conv2d_full(ts_blk.conv2, pt_blk.conv2)
    _inject_bn2d(pt_blk.bn2, ts_blk.bn2)
    if ts_blk.has_downsample and pt_blk.downsample is not None:
        _inject_conv2d_full(ts_blk.ds_conv, pt_blk.downsample[0])
        _inject_bn2d(pt_blk.downsample[1], ts_blk.ds_bn)


def _inject_resnet(pt_enc, ts_enc):
    ch = dict(pt_enc.named_children())
    _inject_conv2d_full(ts_enc.conv1, ch["conv1"])
    _inject_bn2d(ch["bn1"], ts_enc.bn1)
    for stage, ts_stage in [
        ("layer1", ts_enc.layer1),
        ("layer2", ts_enc.layer2),
        ("layer3", ts_enc.layer3),
        ("layer4", ts_enc.layer4),
    ]:
        for pt_b, ts_b in zip(ch[stage], ts_stage):
            _inject_basic_block(pt_b, ts_b)


def inject_backbone(pt_bb, ts_bb):
    _inject_resnet(pt_bb.image_encoder, ts_bb.image_encoder)
    _inject_resnet(pt_bb.lidar_encoder, ts_bb.lidar_encoder)

    for i in range(4):
        _inject_gpt(pt_bb.transformers[i], ts_bb.transformers[i])

    for i in range(4):
        # Channel-conversion convs have paired external F.Bias — weight only into Conv2d
        getattr(ts_bb, f"lidar_channel_to_img_{i}").params[0][1].data = (
            pt_bb.lidar_channel_to_img[i].weight.data.cpu().numpy()
        )
        getattr(ts_bb, f"lidar_channel_to_img_bias_{i}").params[0][1].data = (
            pt_bb.lidar_channel_to_img[i]
            .bias.data.cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )
        getattr(ts_bb, f"img_channel_to_lidar_{i}").params[0][1].data = (
            pt_bb.img_channel_to_lidar[i].weight.data.cpu().numpy()
        )
        getattr(ts_bb, f"img_channel_to_lidar_bias_{i}").params[0][1].data = (
            pt_bb.img_channel_to_lidar[i]
            .bias.data.cpu()
            .numpy()
            .reshape(1, -1, 1, 1)
            .astype(np.float32)
        )

    if hasattr(pt_bb, "lidar_to_img_features_end"):
        ts_bb.lidar_to_img_features_end.params[0][1].data = (
            pt_bb.lidar_to_img_features_end.weight.data.cpu()
            .numpy()
            .T.astype(np.float32)
        )
        ts_bb.lidar_to_img_features_end_bias.params[0][1].data = (
            pt_bb.lidar_to_img_features_end.bias.data.cpu().numpy().astype(np.float32)
        )

    # FPN convs: weight-only into Conv2d; bias goes to external F.Bias op
    for attr in ("c5_conv", "up_conv5", "up_conv4"):
        if hasattr(pt_bb, attr):
            pt_conv = getattr(pt_bb, attr)
            _inject_conv2d_weight_only(getattr(ts_bb, attr), pt_conv)
            _inject_conv2d_bias_op(
                getattr(ts_bb, f"{attr}_bias"), pt_conv, reshape=(1, -1, 1, 1)
            )


# ---------------------------------------------------------------------------
# Standard TransformerDecoder
# ---------------------------------------------------------------------------


def inject_std_decoder_layer(ts_layer, pt_layer, d_model):
    _inject_mha(
        ts_layer.self_attn_query,
        ts_layer.self_attn_query_bias,
        ts_layer.self_attn_key,
        ts_layer.self_attn_key_bias,
        ts_layer.self_attn_value,
        ts_layer.self_attn_value_bias,
        ts_layer.self_attn_out,
        ts_layer.self_attn_out_bias,
        pt_layer.self_attn,
        d_model,
    )
    _inject_mha(
        ts_layer.cross_attn_query,
        ts_layer.cross_attn_query_bias,
        ts_layer.cross_attn_key,
        ts_layer.cross_attn_key_bias,
        ts_layer.cross_attn_value,
        ts_layer.cross_attn_value_bias,
        ts_layer.cross_attn_out,
        ts_layer.cross_attn_out_bias,
        pt_layer.multihead_attn,
        d_model,
    )
    _inject_linear(ts_layer.ffn_linear1, ts_layer.ffn_bias1, pt_layer.linear1)
    _inject_linear(ts_layer.ffn_linear2, ts_layer.ffn_bias2, pt_layer.linear2)
    _inject_layernorm(ts_layer.norm1, pt_layer.norm1)
    _inject_layernorm(ts_layer.norm2, pt_layer.norm2)
    _inject_layernorm(ts_layer.norm3, pt_layer.norm3)


# ---------------------------------------------------------------------------
# AgentHead
# ---------------------------------------------------------------------------


def inject_agent_head(ts_head, pt_head):
    _inject_linear(
        ts_head._mlp_states_linear1, ts_head._mlp_states_bias1, pt_head._mlp_states[0]
    )
    _inject_linear(
        ts_head._mlp_states_linear2, ts_head._mlp_states_bias2, pt_head._mlp_states[2]
    )
    _inject_linear(
        ts_head._mlp_label_linear, ts_head._mlp_label_bias, pt_head._mlp_label[0]
    )


# ---------------------------------------------------------------------------
# TrajectoryHead (CustomTransformerDecoder layers)
# ---------------------------------------------------------------------------


def inject_custom_decoder_layer(ts_layer, pt_layer):
    # BEV cross-attention
    _inject_linear(
        ts_layer.cross_bev_attention.attention_weights_linear,
        ts_layer.cross_bev_attention.attention_weights_bias,
        pt_layer.cross_bev_attention.attention_weights,
    )
    _inject_linear(
        ts_layer.cross_bev_attention.output_proj_linear,
        ts_layer.cross_bev_attention.output_proj_bias,
        pt_layer.cross_bev_attention.output_proj,
    )
    pt_vp = pt_layer.cross_bev_attention.value_proj[0]
    _inject_conv2d_weight_only(ts_layer.cross_bev_attention.value_proj_conv, pt_vp)
    _inject_conv2d_bias_op(
        ts_layer.cross_bev_attention.value_proj_bias, pt_vp, reshape=(1, 256, 1, 1)
    )

    # Agent + ego cross-attentions
    _inject_mha(
        ts_layer.cross_agent_q_linear,
        ts_layer.cross_agent_q_bias,
        ts_layer.cross_agent_k_linear,
        ts_layer.cross_agent_k_bias,
        ts_layer.cross_agent_v_linear,
        ts_layer.cross_agent_v_bias,
        ts_layer.cross_agent_out_proj,
        ts_layer.cross_agent_out_bias,
        pt_layer.cross_agent_attention,
        ts_layer.d_model,
    )
    _inject_mha(
        ts_layer.cross_ego_q_linear,
        ts_layer.cross_ego_q_bias,
        ts_layer.cross_ego_k_linear,
        ts_layer.cross_ego_k_bias,
        ts_layer.cross_ego_v_linear,
        ts_layer.cross_ego_v_bias,
        ts_layer.cross_ego_out_proj,
        ts_layer.cross_ego_out_bias,
        pt_layer.cross_ego_attention,
        ts_layer.d_model,
    )

    # FFN + norms
    _inject_linear(ts_layer.ffn_linear1, ts_layer.ffn_bias1, pt_layer.ffn[0])
    _inject_linear(ts_layer.ffn_linear2, ts_layer.ffn_bias2, pt_layer.ffn[2])
    _inject_layernorm(ts_layer.norm1, pt_layer.norm1)
    _inject_layernorm(ts_layer.norm2, pt_layer.norm2)
    _inject_layernorm(ts_layer.norm3, pt_layer.norm3)

    # Time modulation
    _inject_linear(
        ts_layer.time_modulation.scale_shift_linear,
        ts_layer.time_modulation.scale_shift_bias,
        pt_layer.time_modulation.scale_shift_mlp[1],
    )

    # DiffMotionPlanningRefinementModule
    # plan_cls_branch: [Linear, ReLU, LayerNorm, Linear, ReLU, LayerNorm, Linear]
    _inject_linear(
        ts_layer.task_decoder.plan_cls_linear1,
        ts_layer.task_decoder.plan_cls_bias1,
        pt_layer.task_decoder.plan_cls_branch[0],
    )
    _inject_layernorm(
        ts_layer.task_decoder.plan_cls_ln1, pt_layer.task_decoder.plan_cls_branch[2]
    )
    _inject_linear(
        ts_layer.task_decoder.plan_cls_linear2,
        ts_layer.task_decoder.plan_cls_bias2,
        pt_layer.task_decoder.plan_cls_branch[3],
    )
    _inject_layernorm(
        ts_layer.task_decoder.plan_cls_ln2, pt_layer.task_decoder.plan_cls_branch[5]
    )
    _inject_linear(
        ts_layer.task_decoder.plan_cls_linear3,
        ts_layer.task_decoder.plan_cls_bias3,
        pt_layer.task_decoder.plan_cls_branch[6],
    )
    # plan_reg_branch: [Linear, ReLU, Linear, ReLU, Linear]
    _inject_linear(
        ts_layer.task_decoder.plan_reg_linear1,
        ts_layer.task_decoder.plan_reg_bias1,
        pt_layer.task_decoder.plan_reg_branch[0],
    )
    _inject_linear(
        ts_layer.task_decoder.plan_reg_linear2,
        ts_layer.task_decoder.plan_reg_bias2,
        pt_layer.task_decoder.plan_reg_branch[2],
    )
    _inject_linear(
        ts_layer.task_decoder.plan_reg_linear3,
        ts_layer.task_decoder.plan_reg_bias3,
        pt_layer.task_decoder.plan_reg_branch[4],
    )


def inject_trajectory_head(ts_head, pt_head):
    ts_head.plan_anchor.data = pt_head.plan_anchor.data.cpu().numpy().astype(np.float32)
    for it in range(2):
        p = f"iter{it}"
        # Plan anchor encoder: [Linear(512,256), ReLU, LayerNorm(256), Linear(256,256)]
        _inject_linear(
            getattr(ts_head, f"plan_enc_linear1_{p}"),
            getattr(ts_head, f"plan_enc_bias1_{p}"),
            pt_head.plan_anchor_encoder[0],
        )
        _inject_layernorm(
            getattr(ts_head, f"plan_enc_ln_{p}"), pt_head.plan_anchor_encoder[2]
        )
        _inject_linear(
            getattr(ts_head, f"plan_enc_linear2_{p}"),
            getattr(ts_head, f"plan_enc_bias2_{p}"),
            pt_head.plan_anchor_encoder[3],
        )
        # Time MLP: [SinusoidalPosEmb, Linear, Mish, Linear]
        _inject_linear(
            getattr(ts_head, f"time_linear1_{p}"),
            getattr(ts_head, f"time_bias1_{p}"),
            pt_head.time_mlp[1],
        )
        _inject_linear(
            getattr(ts_head, f"time_linear2_{p}"),
            getattr(ts_head, f"time_bias2_{p}"),
            pt_head.time_mlp[3],
        )
        ts_decoder = getattr(ts_head, f"diff_decoder_{p}")
        for ts_l, pt_l in zip(ts_decoder.layers, pt_head.diff_decoder.layers):
            inject_custom_decoder_layer(ts_l, pt_l)


# ---------------------------------------------------------------------------
# Full model injection
# ---------------------------------------------------------------------------


def inject_all_weights(ts_model, pt_model):
    d_model = pt_model._config.tf_d_model

    # 1. Backbone
    inject_backbone(pt_model._backbone, ts_model._backbone)

    # 2. Embeddings
    _inject_embedding(ts_model._keyval_embedding, pt_model._keyval_embedding)
    _inject_embedding(ts_model._query_embedding, pt_model._query_embedding)

    # 3. BEV downscale — weight only; bias lives in _bev_downscale_bias F.Bias op
    _inject_conv2d_weight_only(ts_model._bev_downscale, pt_model._bev_downscale)
    _inject_conv2d_bias_op(
        ts_model._bev_downscale_bias, pt_model._bev_downscale, reshape=(1, -1, 1, 1)
    )

    # 4. Status encoding
    _inject_linear(
        ts_model._status_encoding,
        ts_model._status_encoding_bias,
        pt_model._status_encoding,
    )

    # 5. BEV semantic head — weight only for each conv; bias in paired F.Bias ops
    _inject_conv2d_weight_only(
        ts_model._bev_semantic_conv1, pt_model._bev_semantic_head[0]
    )
    _inject_conv2d_bias_op(
        ts_model._bev_semantic_conv1_bias,
        pt_model._bev_semantic_head[0],
        reshape=(1, -1, 1, 1),
    )
    _inject_conv2d_weight_only(
        ts_model._bev_semantic_conv2, pt_model._bev_semantic_head[2]
    )
    _inject_conv2d_bias_op(
        ts_model._bev_semantic_conv2_bias,
        pt_model._bev_semantic_head[2],
        reshape=(1, -1, 1, 1),
    )

    # 6. BEV projection: [Linear(320→256), ReLU, LayerNorm(256)]
    _inject_linear(
        ts_model.bev_proj_linear1, ts_model.bev_proj_bias1, pt_model.bev_proj[0]
    )
    _inject_layernorm(ts_model.bev_proj_ln, pt_model.bev_proj[2])

    # 7. Standard TransformerDecoder
    for ts_l, pt_l in zip(ts_model._tf_decoder_layers, pt_model._tf_decoder.layers):
        inject_std_decoder_layer(ts_l, pt_l, d_model)

    # 8. Agent head
    inject_agent_head(ts_model._agent_head, pt_model._agent_head)

    # 9. Trajectory head
    inject_trajectory_head(ts_model._trajectory_head, pt_model._trajectory_head)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("V2TransfuserModel Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)

    # Minimal config for fast testing
    config = TransfuserConfig()
    config.image_architecture = "resnet18"
    config.lidar_architecture = "resnet18"
    config.camera_width = 32
    config.camera_height = 32
    config.lidar_resolution_width = 32
    config.lidar_resolution_height = 32
    config.img_vert_anchors = 1
    config.img_horz_anchors = 1
    config.lidar_vert_anchors = 1
    config.lidar_horz_anchors = 1
    config.bev_pixel_width = 16
    config.bev_pixel_height = 8
    config.num_bounding_boxes = 3
    batch_size = 1

    plan_anchor_path = os.path.join(
        os.path.dirname(__file__), "test_plan_anchor_v8.npy"
    )
    np.save(plan_anchor_path, np.random.randn(20, 8, 2).astype(np.float32))
    config.plan_anchor_path = plan_anchor_path

    # ------------------------------------------------------------------
    # Create models
    # ------------------------------------------------------------------
    print("\n--- Creating PyTorch Model ---")
    try:
        model_pytorch = V2TransfuserModel_PyTorch(config)
        model_pytorch.eval()
        print("PASS: PyTorch V2TransfuserModel created")
    except Exception as e:
        import traceback

        traceback.print_exc()
        _cleanup(plan_anchor_path)
        return

    print("\n--- Creating TTSIM Model ---")
    try:
        model_ttsim = V2TransfuserModel_TTSIM(config)
        print("PASS: TTSIM V2TransfuserModel created")
    except Exception as e:
        import traceback

        traceback.print_exc()
        _cleanup(plan_anchor_path)
        return

    # ------------------------------------------------------------------
    # Weight injection
    # ------------------------------------------------------------------
    print("\n--- Injecting Weights ---")
    try:
        inject_all_weights(model_ttsim, model_pytorch)
        print("PASS: Weight injection complete")
    except Exception as e:
        import traceback

        traceback.print_exc()
        _cleanup(plan_anchor_path)
        return

    # ------------------------------------------------------------------
    # Generate inputs
    # ------------------------------------------------------------------
    lidar_ch = (
        2 * config.lidar_seq_len if config.use_ground_plane else config.lidar_seq_len
    )
    camera_data = np.random.randn(
        batch_size, 3, config.camera_height, config.camera_width
    ).astype(np.float32)
    lidar_data = np.random.randn(
        batch_size,
        lidar_ch,
        config.lidar_resolution_height,
        config.lidar_resolution_width,
    ).astype(np.float32)
    status_data = np.random.randn(batch_size, 8).astype(np.float32)

    print(f"\n--- Input Shapes ---")
    print(f"  Camera: {list(camera_data.shape)}")
    print(f"  LiDAR:  {list(lidar_data.shape)}")
    print(f"  Status: {list(status_data.shape)}")

    # ------------------------------------------------------------------
    # PyTorch forward pass
    # ------------------------------------------------------------------
    print("\n--- PyTorch Forward Pass ---")
    camera_torch = torch.from_numpy(camera_data)
    lidar_torch = torch.from_numpy(lidar_data)
    status_torch = torch.from_numpy(status_data)
    features_pt = {
        "camera_feature": camera_torch,
        "lidar_feature": lidar_torch,
        "status_feature": status_torch,
    }
    try:
        with torch.no_grad():
            output_pytorch = model_pytorch(features_pt, targets=None)
        print("PASS: PyTorch forward pass complete")
        for key, val in output_pytorch.items():
            if isinstance(val, torch.Tensor):
                arr = val.cpu().numpy()
                print(
                    f"  {key}: shape={list(arr.shape)}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
                )
    except Exception as e:
        import traceback

        traceback.print_exc()
        _cleanup(plan_anchor_path)
        return

    # ------------------------------------------------------------------
    # TTSIM forward pass
    # ------------------------------------------------------------------
    print("\n--- TTSIM Forward Pass ---")
    camera_ttsim = F._from_data("camera_feature", camera_data)
    lidar_ttsim = F._from_data("lidar_feature", lidar_data)
    status_ttsim = F._from_data("status_feature", status_data)
    camera_ttsim.link_module = model_ttsim
    lidar_ttsim.link_module = model_ttsim
    status_ttsim.link_module = model_ttsim
    features_ttsim = {
        "camera_feature": camera_ttsim,
        "lidar_feature": lidar_ttsim,
        "status_feature": status_ttsim,
    }
    try:
        output_ttsim = model_ttsim(features_ttsim, targets=None)
        print("PASS: TTSIM forward pass complete")
        for key, val in output_ttsim.items():
            if val.data is not None:
                arr = val.data
                print(
                    f"  {key}: shape={list(arr.shape)}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}"
                )
            else:
                print(f"  {key}: shape={list(val.shape)} (data=None)")
    except Exception as e:
        import traceback

        traceback.print_exc()
        _cleanup(plan_anchor_path)
        return

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------
    print("\n--- Shape Validation ---")
    expected_shapes = {
        "bev_semantic_map": [
            batch_size,
            config.num_bev_classes,
            config.lidar_resolution_height // 2,
            config.lidar_resolution_width,
        ],
        "agent_states": [batch_size, config.num_bounding_boxes, 5],
        "agent_labels": [batch_size, config.num_bounding_boxes],
        "trajectory": [batch_size, config.trajectory_sampling.num_poses, 3],
    }
    shape_ok = True
    for key, exp in expected_shapes.items():
        pt_val = output_pytorch.get(key)
        ts_val = output_ttsim.get(key)
        pt_shape = list(pt_val.shape) if pt_val is not None else None
        ts_shape = (
            list(ts_val.data.shape)
            if (ts_val is not None and ts_val.data is not None)
            else None
        )
        if ts_shape and key == "agent_labels" and ts_shape == exp + [1]:
            ts_shape = exp  # TTSIM returns [B,N,1], compare after squeeze
        match = (pt_shape == exp) and (ts_shape == exp)
        status = "PASS" if match else "FAIL"
        print(
            f"  {key}: expected={exp}  pytorch={pt_shape}  ttsim={ts_shape}  {status}"
        )
        if not match:
            shape_ok = False

    # ------------------------------------------------------------------
    # Numerical comparison
    # ------------------------------------------------------------------
    print("\n--- Numerical Comparison ---")
    atol = 0.0001
    rtol = 0.0001
    print(f"  Tolerance: atol={atol}, rtol={rtol}")

    compare_keys = ["bev_semantic_map", "agent_states", "agent_labels"]
    results = {}
    for key in compare_keys:
        pt_val = output_pytorch.get(key)
        ts_val = output_ttsim.get(key)
        if pt_val is None or ts_val is None or ts_val.data is None:
            print(f"\n  {key}: SKIP")
            continue
        pt_np = pt_val.cpu().numpy() if isinstance(pt_val, torch.Tensor) else pt_val
        ts_np = ts_val.data
        if (
            key == "agent_labels"
            and ts_np.ndim == pt_np.ndim + 1
            and ts_np.shape[-1] == 1
        ):
            ts_np = ts_np.squeeze(-1)

        diff = np.abs(pt_np - ts_np)
        max_diff = diff.max()
        mean_diff = diff.mean()
        passed = bool(np.allclose(pt_np, ts_np, atol=atol, rtol=rtol))
        results[key] = passed

        print(f"\n  {key}:")
        print(f"    PyTorch shape: {list(pt_np.shape)}")
        print(f"    TTSIM shape:   {list(ts_np.shape)}")
        print(f"    Max absolute difference:  {max_diff:.10f}")
        print(f"    Mean absolute difference: {mean_diff:.10f}")
        print(
            f"    PyTorch stats: min={pt_np.min():.6f}, max={pt_np.max():.6f}, mean={pt_np.mean():.6f}"
        )
        print(
            f"    TTSIM   stats: min={ts_np.min():.6f}, max={ts_np.max():.6f}, mean={ts_np.mean():.6f}"
        )
        if passed:
            print(f"    PASS")
        else:
            print(f"    FAIL")
            flat = diff.flatten()
            top5 = np.argsort(flat)[-5:][::-1]
            print(f"    Top-5 mismatches:")
            for idx in top5:
                coord = np.unravel_index(idx, diff.shape)
                print(
                    f"      {coord}: pytorch={pt_np[coord]:.6f}  ttsim={ts_np[coord]:.6f}  diff={flat[idx]:.6f}"
                )

    # Trajectory (informational only)
    print(f"\n  trajectory:")
    pt_traj = output_pytorch.get("trajectory")
    ts_traj = output_ttsim.get("trajectory")
    if pt_traj is not None:
        print(f"    PyTorch shape: {list(pt_traj.shape)}")
    if ts_traj is not None and ts_traj.data is not None:
        print(f"    TTSIM   shape: {list(ts_traj.data.shape)}")
    print(f"    (Comparison skipped: TTSIM uses mock scheduler)")
    print(f"    (Trajectory validated separately in test_6_trajectory_head.py)")

    # ------------------------------------------------------------------
    # Overall result
    # ------------------------------------------------------------------
    all_pass = shape_ok and results and all(results.values())
    print("\n" + "=" * 70)
    if all_pass:
        print(f"OVERALL: PASS - All outputs match (atol={atol}, rtol={rtol})")
    else:
        print(f"OVERALL: FAIL - Some outputs don't match (atol={atol}, rtol={rtol})")
    print(f"  Shape validation: {'PASS' if shape_ok else 'FAIL'}")
    for key, passed in results.items():
        print(f"  {key}: {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    _cleanup(plan_anchor_path)
    print()


def _cleanup(path):
    if os.path.exists(path):
        os.remove(path)


if __name__ == "__main__":
    main()
