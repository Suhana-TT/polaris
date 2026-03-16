# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# Standard path logic
current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.sim_nn as SimNN

class TtsimSegformerMLP:
    """
    Linear Embedding used in the Decode Head to unify channel dimensions.
    """
    def __init__(self, name: str, parameters):
        self.name = name
        
        in_feat = parameters["proj"]["weight"].shape[1]
        out_feat = parameters["proj"]["weight"].shape[0]

        # Initialize the graph node (shapes only)
        self.proj = SimNN.Linear(
            name=f"{self.name}_proj",
            in_features=in_feat,
            out_features=out_feat,
            bias=True
        )

    def __call__(self, hidden_states):
        # The DecodeHead passes a flattened 3D sequence here, 
        # so we just pipe it straight into the Linear layer!
        hidden_states = self.proj(hidden_states)

        return hidden_states
