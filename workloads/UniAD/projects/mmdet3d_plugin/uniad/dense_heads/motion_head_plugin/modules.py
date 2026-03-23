# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim: motion_head_plugin/modules.py — SimNN replacements.
No torch, no mmcv, no einops imports.
"""

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from ....ttsim_utils import TRANSFORMER_LAYER_SEQUENCE  # type: ignore[import-not-found]


class IntentionInteraction(SimNN.Module):
    """TTSim SimNN module for modeling interaction between anchors."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.name = "intention_interaction"
        self.embed_dims = embed_dims

        # Self-attention components (encoder-style)
        self.wq = SimNN.Linear(self.name + ".wq", embed_dims, embed_dims)
        self.wk = SimNN.Linear(self.name + ".wk", embed_dims, embed_dims)
        self.wv = SimNN.Linear(self.name + ".wv", embed_dims, embed_dims)
        self.wo = SimNN.Linear(self.name + ".wo", embed_dims, embed_dims)
        self.sm = F.Softmax(self.name + ".sm", axis=-1)
        self.dropout = F.Dropout(self.name + ".dropout", dropout, True)

        # FFN
        self.fc1 = SimNN.Linear(self.name + ".fc1", embed_dims, embed_dims * 2)
        self.relu = F.Relu(self.name + ".relu")
        self.fc2 = SimNN.Linear(self.name + ".fc2", embed_dims * 2, embed_dims)

        # Norms
        self.norm1 = F.LayerNorm(self.name + ".norm1", embed_dims)
        self.norm2 = F.LayerNorm(self.name + ".norm2", embed_dims)

        self.add1 = F.Add(self.name + ".add1")
        self.add2 = F.Add(self.name + ".add2")

        super().link_op2module()

    def __call__(self, query):
        # query: (B, A, P, D) — self-attention over P modes per agent
        q = self.wq(query)
        k = self.wk(query)
        v = self.wv(query)
        attn = self.sm(q)
        attn_out = self.dropout(self.wo(self.sm(attn)))
        query = self.norm1(self.add1(query, attn_out))
        ffn_out = self.dropout(self.fc2(self.relu(self.fc1(query))))
        query = self.norm2(self.add2(query, ffn_out))
        return query


class TrackAgentInteraction(SimNN.Module):
    """TTSim SimNN module for modeling interaction between agents."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.name = "track_agent_interaction"
        self.embed_dims = embed_dims

        # Cross-attention components
        self.wq = SimNN.Linear(self.name + ".wq", embed_dims, embed_dims)
        self.wk = SimNN.Linear(self.name + ".wk", embed_dims, embed_dims)
        self.wv = SimNN.Linear(self.name + ".wv", embed_dims, embed_dims)
        self.wo = SimNN.Linear(self.name + ".wo", embed_dims, embed_dims)
        self.sm = F.Softmax(self.name + ".sm", axis=-1)
        self.dropout = F.Dropout(self.name + ".dropout", dropout, True)

        # Self-attention components
        self.sq = SimNN.Linear(self.name + ".sq", embed_dims, embed_dims)
        self.sk = SimNN.Linear(self.name + ".sk", embed_dims, embed_dims)
        self.sv = SimNN.Linear(self.name + ".sv", embed_dims, embed_dims)
        self.so = SimNN.Linear(self.name + ".so", embed_dims, embed_dims)
        self.ssm = F.Softmax(self.name + ".ssm", axis=-1)

        # FFN
        self.fc1 = SimNN.Linear(self.name + ".fc1", embed_dims, embed_dims * 2)
        self.relu = F.Relu(self.name + ".relu")
        self.fc2 = SimNN.Linear(self.name + ".fc2", embed_dims * 2, embed_dims)

        # Norms
        self.norm1 = F.LayerNorm(self.name + ".norm1", embed_dims)
        self.norm2 = F.LayerNorm(self.name + ".norm2", embed_dims)
        self.norm3 = F.LayerNorm(self.name + ".norm3", embed_dims)

        self.add1 = F.Add(self.name + ".add1")
        self.add2 = F.Add(self.name + ".add2")
        self.add3 = F.Add(self.name + ".add3")

        super().link_op2module()

    def __call__(self, query, key, query_pos=None, key_pos=None):
        # query: (B, A, P, D), key: (B, A, D)
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(key)
        cross_out = self.dropout(self.wo(self.sm(q)))
        query = self.norm1(self.add1(query, cross_out))
        ffn_out = self.dropout(self.fc2(self.relu(self.fc1(query))))
        query = self.norm2(self.add2(query, ffn_out))
        return query


class MapInteraction(SimNN.Module):
    """TTSim SimNN module for modeling interaction between agents and map."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.name = "map_interaction"
        self.embed_dims = embed_dims

        # Cross-attention components
        self.wq = SimNN.Linear(self.name + ".wq", embed_dims, embed_dims)
        self.wk = SimNN.Linear(self.name + ".wk", embed_dims, embed_dims)
        self.wv = SimNN.Linear(self.name + ".wv", embed_dims, embed_dims)
        self.wo = SimNN.Linear(self.name + ".wo", embed_dims, embed_dims)
        self.sm = F.Softmax(self.name + ".sm", axis=-1)
        self.dropout = F.Dropout(self.name + ".dropout", dropout, True)

        # FFN
        self.fc1 = SimNN.Linear(self.name + ".fc1", embed_dims, embed_dims * 2)
        self.relu = F.Relu(self.name + ".relu")
        self.fc2 = SimNN.Linear(self.name + ".fc2", embed_dims * 2, embed_dims)

        # Norms
        self.norm1 = F.LayerNorm(self.name + ".norm1", embed_dims)
        self.norm2 = F.LayerNorm(self.name + ".norm2", embed_dims)

        self.add1 = F.Add(self.name + ".add1")
        self.add2 = F.Add(self.name + ".add2")

        super().link_op2module()

    def __call__(self, query, key, query_pos=None, key_pos=None):
        # query: (B, A, P, D), key: (B, M, D)
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(key)
        cross_out = self.dropout(self.wo(self.sm(q)))
        query = self.norm1(self.add1(query, cross_out))
        ffn_out = self.dropout(self.fc2(self.relu(self.fc1(query))))
        query = self.norm2(self.add2(query, ffn_out))
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MotionTransformerDecoder(SimNN.Module):
    """TTSim SimNN motion transformer decoder."""

    def __init__(
        self,
        pc_range=None,
        embed_dims=256,
        transformerlayers=None,
        num_layers=3,
        **kwargs,
    ):
        super().__init__()
        self.name = "motion_transformer_decoder"
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers

        self.intention_interaction_layers = IntentionInteraction(embed_dims=embed_dims)
        self.intention_interaction_layers.name = self.name + ".intention_interaction"

        self._track_agent_layers = []
        self._map_layers = []
        for i in range(num_layers):
            ta = TrackAgentInteraction(embed_dims=embed_dims)
            ta.name = self.name + f".track_agent{i}"
            self._track_agent_layers.append(ta)
            setattr(self, f"track_agent{i}", ta)

            mi = MapInteraction(embed_dims=embed_dims)
            mi.name = self.name + f".map_interaction{i}"
            self._map_layers.append(mi)
            setattr(self, f"map_interaction{i}", mi)

        # Fuser MLPs
        self.static_dynamic_fuser_l1 = SimNN.Linear(
            self.name + ".static_dynamic_fuser.l1", embed_dims * 2, embed_dims * 2
        )
        self.static_dynamic_fuser_relu = F.Relu(
            self.name + ".static_dynamic_fuser.relu"
        )
        self.static_dynamic_fuser_l2 = SimNN.Linear(
            self.name + ".static_dynamic_fuser.l2", embed_dims * 2, embed_dims
        )

        self.dynamic_embed_fuser_l1 = SimNN.Linear(
            self.name + ".dynamic_embed_fuser.l1", embed_dims * 3, embed_dims * 2
        )
        self.dynamic_embed_fuser_relu = F.Relu(self.name + ".dynamic_embed_fuser.relu")
        self.dynamic_embed_fuser_l2 = SimNN.Linear(
            self.name + ".dynamic_embed_fuser.l2", embed_dims * 2, embed_dims
        )

        self.in_query_fuser_l1 = SimNN.Linear(
            self.name + ".in_query_fuser.l1", embed_dims * 2, embed_dims * 2
        )
        self.in_query_fuser_relu = F.Relu(self.name + ".in_query_fuser.relu")
        self.in_query_fuser_l2 = SimNN.Linear(
            self.name + ".in_query_fuser.l2", embed_dims * 2, embed_dims
        )

        self.out_query_fuser_l1 = SimNN.Linear(
            self.name + ".out_query_fuser.l1", embed_dims * 4, embed_dims * 2
        )
        self.out_query_fuser_relu = F.Relu(self.name + ".out_query_fuser.relu")
        self.out_query_fuser_l2 = SimNN.Linear(
            self.name + ".out_query_fuser.l2", embed_dims * 2, embed_dims
        )

        self.cat_static_dynamic = F.ConcatX(self.name + ".cat_static_dynamic", axis=-1)
        self.cat_dynamic_embed = F.ConcatX(self.name + ".cat_dynamic_embed", axis=-1)
        self.cat_in_query = F.ConcatX(self.name + ".cat_in_query", axis=-1)
        self.cat_out_query = F.ConcatX(self.name + ".cat_out_query", axis=-1)

        super().link_op2module()

    def __call__(
        self,
        track_query,
        lane_query,
        track_query_pos=None,
        lane_query_pos=None,
        track_bbox_results=None,
        bev_embed=None,
        reference_trajs=None,
        traj_reg_branches=None,
        agent_level_embedding=None,
        scene_level_ego_embedding=None,
        scene_level_offset_embedding=None,
        learnable_embed=None,
        agent_level_embedding_layer=None,
        scene_level_ego_embedding_layer=None,
        scene_level_offset_embedding_layer=None,
        **kwargs,
    ):
        raise NotImplementedError(
            "MotionTransformerDecoder.__call__: full forward requires runtime tensor shapes; "
            "use ttsim simulation path from motion_head.py"
        )
