# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import json


class SegformerConfig:
    """Config class matching tt-metal structure"""
    
    def __init__(self, config_dict):
        # Model architecture
        self.num_channels = config_dict.get("num_channels", 3)
        self.num_encoder_blocks = config_dict.get("num_encoder_blocks", 4)
        self.depths = config_dict.get("depths", [2, 2, 2, 2])
        self.sr_ratios = config_dict.get("sr_ratios", [8, 4, 2, 1])
        self.hidden_sizes = config_dict.get("hidden_sizes", [32, 64, 160, 256])
        self.patch_sizes = config_dict.get("patch_sizes", [7, 3, 3, 3])
        self.strides = config_dict.get("strides", [4, 2, 2, 2])
        self.num_attention_heads = config_dict.get("num_attention_heads", [1, 2, 5, 8])
        self.mlp_ratios = config_dict.get("mlp_ratios", [4, 4, 4, 4])
        
        # Output settings
        self.output_attentions = config_dict.get("output_attentions", False)
        self.output_hidden_states = config_dict.get("output_hidden_states", False)
        self.use_return_dict = config_dict.get("return_dict", True)
        
        # Other model settings
        self.hidden_act = config_dict.get("hidden_act", "gelu")
        self.hidden_dropout_prob = config_dict.get("hidden_dropout_prob", 0.0)
        self.attention_probs_dropout_prob = config_dict.get("attention_probs_dropout_prob", 0.0)
        self.classifier_dropout_prob = config_dict.get("classifier_dropout_prob", 0.1)
        self.initializer_range = config_dict.get("initializer_range", 0.02)
        self.drop_path_rate = config_dict.get("drop_path_rate", 0.1)
        self.layer_norm_eps = config_dict.get("layer_norm_eps", 1e-6)
        self.decoder_hidden_size = config_dict.get("decoder_hidden_size", 256)
        
        # Image settings
        self.image_size = config_dict.get("image_size", 224)
        self.reshape_last_stage = config_dict.get("reshape_last_stage", True)
        
        # Semantic segmentation settings
        self.semantic_loss_ignore_index = config_dict.get("semantic_loss_ignore_index", 255)
        self.downsampling_rates = config_dict.get("downsampling_rates", [1, 4, 8, 16])
        
        # Label mappings
        self.id2label = config_dict.get("id2label", {})
        self.label2id = config_dict.get("label2id", {})
        self.num_labels = len(self.id2label) if self.id2label else 150


def load_config(config_path):
    """Load config from JSON file (like tt-metal)"""
    
    # Get base directory (workloads/segformer/)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, config_path)
    
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            config_dict = json.load(f)
        print(f"[INFO] Loaded config from: {full_path}")
    else:
        print(f"[WARNING] Config file not found at {full_path}, using defaults")
        config_dict = {}
    
    return SegformerConfig(config_dict)