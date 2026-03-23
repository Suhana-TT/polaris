# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: motion_head_plugin/base_motion_head.py — SimNN replacement.
No torch, no mmcv, no mmdet imports.
"""

import copy
import pickle
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from ....ttsim_utils import build_loss, build_transformer_layer_sequence  # type: ignore[import-not-found]


class BaseMotionHead(SimNN.Module):
    # Attributes set by subclass before calling _build_layers / _init_layers
    embed_dims: int
    num_anchor: int
    num_anchor_group: int
    num_reg_fcs: int
    predict_steps: int

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.name = "base_motion_head"

    def _build_loss(self, loss_traj):
        """
        Build the loss function for the motion prediction task.
        Training-only: raises NotImplementedError in simulation.
        """
        raise NotImplementedError("_build_loss: training-only")

    def _load_anchors(self, anchor_info_path):
        """
        Load the anchor information from a file.
        """
        anchor_infos = pickle.load(open(anchor_info_path, "rb"))
        self.kmeans_anchors = np.stack(
            [np.asarray(a) for a in anchor_infos["anchors_all"]]
        )  # Nc, Pc, steps, 2

    def _build_layers(self, transformerlayers, det_layer_num):
        """
        Build the layers of the motion prediction module.
        """
        self.learnable_motion_query_embedding = F.Embedding(
            self.name + ".motion_query_emb",
            self.num_anchor * self.num_anchor_group,
            self.embed_dims,
        )

        # motionformer: stub — full implementation in motion_head.py
        self.motionformer = None  # set by subclass via _build_motionformer

        # Track query fuser: linear -> layer_norm -> relu
        self.layer_track_query_fuser_linear = SimNN.Linear(
            self.name + ".track_query_fuser.linear",
            self.embed_dims * det_layer_num,
            self.embed_dims,
        )
        self.layer_track_query_fuser_norm = F.LayerNorm(
            self.name + ".track_query_fuser.norm", self.embed_dims
        )
        self.layer_track_query_fuser_relu = F.Relu(
            self.name + ".track_query_fuser.relu"
        )

        # Embedding layers
        self.agent_level_embedding_layer_l1 = SimNN.Linear(
            self.name + ".agent_emb.l1", self.embed_dims, self.embed_dims * 2
        )
        self.agent_level_embedding_layer_relu = F.Relu(self.name + ".agent_emb.relu")
        self.agent_level_embedding_layer_l2 = SimNN.Linear(
            self.name + ".agent_emb.l2", self.embed_dims * 2, self.embed_dims
        )

        self.scene_level_ego_embedding_layer_l1 = SimNN.Linear(
            self.name + ".scene_ego_emb.l1", self.embed_dims, self.embed_dims * 2
        )
        self.scene_level_ego_embedding_layer_relu = F.Relu(
            self.name + ".scene_ego_emb.relu"
        )
        self.scene_level_ego_embedding_layer_l2 = SimNN.Linear(
            self.name + ".scene_ego_emb.l2", self.embed_dims * 2, self.embed_dims
        )

        self.scene_level_offset_embedding_layer_l1 = SimNN.Linear(
            self.name + ".scene_offset_emb.l1", self.embed_dims, self.embed_dims * 2
        )
        self.scene_level_offset_embedding_layer_relu = F.Relu(
            self.name + ".scene_offset_emb.relu"
        )
        self.scene_level_offset_embedding_layer_l2 = SimNN.Linear(
            self.name + ".scene_offset_emb.l2", self.embed_dims * 2, self.embed_dims
        )

        self.boxes_query_embedding_layer_l1 = SimNN.Linear(
            self.name + ".boxes_query_emb.l1", self.embed_dims, self.embed_dims * 2
        )
        self.boxes_query_embedding_layer_relu = F.Relu(
            self.name + ".boxes_query_emb.relu"
        )
        self.boxes_query_embedding_layer_l2 = SimNN.Linear(
            self.name + ".boxes_query_emb.l2", self.embed_dims * 2, self.embed_dims
        )

        super().link_op2module()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        # traj_cls_branch: Linear -> LayerNorm -> ReLU -> ... -> Linear
        # traj_reg_branch: Linear -> ReLU -> ... -> Linear

        def _make_cls_branch(name_prefix, embed_dims, num_reg_fcs):
            layers = []
            layers.append(SimNN.Linear(name_prefix + ".l0", embed_dims, embed_dims))
            layers.append(F.LayerNorm(name_prefix + ".ln0", embed_dims))
            layers.append(F.Relu(name_prefix + ".relu0"))
            for i in range(num_reg_fcs - 1):
                layers.append(
                    SimNN.Linear(name_prefix + f".l{i+1}", embed_dims, embed_dims)
                )
                layers.append(F.LayerNorm(name_prefix + f".ln{i+1}", embed_dims))
                layers.append(F.Relu(name_prefix + f".relu{i+1}"))
            layers.append(SimNN.Linear(name_prefix + ".out", embed_dims, 1))
            return layers

        def _make_reg_branch(name_prefix, embed_dims, num_reg_fcs, predict_steps):
            layers = []
            layers.append(SimNN.Linear(name_prefix + ".l0", embed_dims, embed_dims))
            layers.append(F.Relu(name_prefix + ".relu0"))
            for i in range(num_reg_fcs - 1):
                layers.append(
                    SimNN.Linear(name_prefix + f".l{i+1}", embed_dims, embed_dims)
                )
                layers.append(F.Relu(name_prefix + f".relu{i+1}"))
            layers.append(
                SimNN.Linear(name_prefix + ".out", embed_dims, predict_steps * 5)
            )
            return layers

        # Store as lists of module handles (not SimNN.Module containers)
        self._traj_cls_branch_template = _make_cls_branch(
            self.name + ".traj_cls", self.embed_dims, self.num_reg_fcs
        )
        self._traj_reg_branch_template = _make_reg_branch(
            self.name + ".traj_reg",
            self.embed_dims,
            self.num_reg_fcs,
            self.predict_steps,
        )

    def _extract_tracking_centers(self, bbox_results, bev_range):
        """
        Extract the bboxes centers and normalize according to the bev range.
        Pure numpy implementation.
        """
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            # gravity_center: shape (N, 3), first two are x, y
            xy = np.asarray(bboxes.gravity_center)[:, :2]
            x_norm = (xy[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
            y_norm = (xy[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(np.stack([x_norm, y_norm], axis=-1))
        return np.stack(det_bbox_posembed)
