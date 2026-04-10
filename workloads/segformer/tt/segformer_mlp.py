# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np

current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.sim_nn as SimNN

class TtsimSegformerMLP:
    """
    Linear embedding used in the SegFormer decode head.
    Input:  [B, S, C]
    Output: [B, S, 256]
    """

    def __init__(self, name: str, parameters):
        self.name = name

        # SimNN.Linear expects param shape [out_features, in_features]
        out_feat = int(parameters["proj"]["weight"].shape[0])
        in_feat = int(parameters["proj"]["weight"].shape[1])

        self.proj = SimNN.Linear(
            name=f"{self.name}_proj",
            in_features=in_feat,
            out_features=out_feat,
            bias=True,
        )

        self.proj.param.data = np.array(parameters["proj"]["weight"], dtype=np.float32).copy()
        self.proj.bias.data = np.array(parameters["proj"]["bias"], dtype=np.float32).reshape(-1).copy()

    def __call__(self, hidden_states):
        return self.proj(hidden_states)