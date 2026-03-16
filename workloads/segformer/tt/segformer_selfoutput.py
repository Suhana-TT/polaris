# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.sim_nn as SimNN

class TtsimSegformerSelfOutput:
    def __init__(self, name: str, hidden_size: int, parameters=None):
        self.name = name
        # We can store parameters here if we ever need to write a custom load_state_dict later
        self.parameters = parameters  
        
        # Initialize the graph node (shapes only)
        self.dense = SimNN.Linear(
            name=f"{self.name}_dense",
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True
        )

    def __call__(self, hidden_states, input_tensor=None):
        # Just pass the data! No parameters keyword needed.
        hidden_states = self.dense(hidden_states)

        return hidden_states
