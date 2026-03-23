# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: track_head_plugin/modules.py — SimNN replacements for MemoryBank and QIM.
No torch, no mmcv imports.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from .track_instance import Instances


class MemoryBank(SimNN.Module):
    """TTSim SimNN module for temporal memory bank."""

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.name = "memory_bank"
        self.save_thresh = args.get("memory_bank_score_thresh", 0.5)
        self.save_period = 3
        self.max_his_length = args.get("memory_bank_len", 4)

        self.save_proj = SimNN.Linear(self.name + ".save_proj", dim_in, dim_in)

        # Temporal attention components
        self.wq = SimNN.Linear(self.name + ".wq", dim_in, dim_in)
        self.wk = SimNN.Linear(self.name + ".wk", dim_in, dim_in)
        self.wv = SimNN.Linear(self.name + ".wv", dim_in, dim_in)
        self.wo = SimNN.Linear(self.name + ".wo", dim_in, dim_in)
        self.sm = F.Softmax(self.name + ".sm", axis=-1)

        self.temporal_fc1 = SimNN.Linear(
            self.name + ".temporal_fc1", dim_in, hidden_dim
        )
        self.temporal_fc2 = SimNN.Linear(
            self.name + ".temporal_fc2", hidden_dim, dim_in
        )
        self.temporal_norm1 = F.LayerNorm(self.name + ".temporal_norm1", dim_in)
        self.temporal_norm2 = F.LayerNorm(self.name + ".temporal_norm2", dim_in)
        self.relu = F.Relu(self.name + ".relu")

        super().link_op2module()

    def update(self, track_instances):
        """No-op in simulation — state update is training-only."""
        pass

    def forward_temporal_attn(self, track_instances):
        """No-op in simulation."""
        return track_instances

    def __call__(self, track_instances: Instances, update_bank=True) -> Instances:
        """No-op forward: memory bank interaction not simulated at op level."""
        return track_instances


class QueryInteractionBase:
    """Base class for query interaction modules."""

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        self.args = args
        self._build_layers(args, dim_in, hidden_dim, dim_out)

    def _build_layers(self, args, dim_in, hidden_dim, dim_out):
        raise NotImplementedError()

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class QueryInteractionModule(SimNN.Module):
    """TTSim SimNN module for query interaction."""

    def __init__(self, args, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.name = "qim"
        self.random_drop = args.get("random_drop", 0.0)
        self.fp_ratio = args.get("fp_ratio", 0.0)
        self.update_query_pos = args.get("update_query_pos", False)

        dropout = args.get("merger_dropout", 0.0)

        # Self-attention
        self.wq = SimNN.Linear(self.name + ".wq", dim_in, dim_in)
        self.wk = SimNN.Linear(self.name + ".wk", dim_in, dim_in)
        self.wv = SimNN.Linear(self.name + ".wv", dim_in, dim_in)
        self.wo = SimNN.Linear(self.name + ".wo", dim_in, dim_in)
        self.sm = F.Softmax(self.name + ".sm", axis=-1)

        self.linear1 = SimNN.Linear(self.name + ".linear1", dim_in, hidden_dim)
        self.linear2 = SimNN.Linear(self.name + ".linear2", hidden_dim, dim_in)
        self.norm1 = F.LayerNorm(self.name + ".norm1", dim_in)
        self.norm2 = F.LayerNorm(self.name + ".norm2", dim_in)
        self.relu = F.Relu(self.name + ".relu")
        self.dropout = F.Dropout(self.name + ".drop", dropout, True)

        if self.update_query_pos:
            self.linear_pos1 = SimNN.Linear(
                self.name + ".linear_pos1", dim_in, hidden_dim
            )
            self.linear_pos2 = SimNN.Linear(
                self.name + ".linear_pos2", hidden_dim, dim_in
            )
            self.norm_pos = F.LayerNorm(self.name + ".norm_pos", dim_in)

        self.linear_feat1 = SimNN.Linear(
            self.name + ".linear_feat1", dim_in, hidden_dim
        )
        self.linear_feat2 = SimNN.Linear(
            self.name + ".linear_feat2", hidden_dim, dim_in
        )
        self.norm_feat = F.LayerNorm(self.name + ".norm_feat", dim_in)

        super().link_op2module()

    def _select_active_tracks(self, data: dict) -> Instances:
        track_instances: Instances = data["track_instances"]
        # inference path: select tracks with valid obj_idxes
        active_track_instances = track_instances[
            np.asarray(track_instances.obj_idxes) >= 0
        ]
        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        """No-op in simulation path — embedding update is training-only."""
        return track_instances

    def __call__(self, data) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        init_track_instances: Instances = data["init_track_instances"]
        merged_track_instances = Instances.cat(
            [init_track_instances, active_track_instances]
        )
        return merged_track_instances
