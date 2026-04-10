# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import numpy as np
from typing import Any, Dict, List

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN


class SegformerMLPBlock(SimNN.Module):
    """MLP block used in decode head"""
    
    def __init__(self, name: str, in_features: int, out_features: int) -> None:
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.proj = SimNN.Linear(f"{name}.proj", in_features, out_features)
        super().link_op2module()
    
    def __call__(self, x: Any) -> Any:
        return self.proj(x)
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        return self.proj.analytical_param_count(lvl + 1)


class SegformerDecodeHead(SimNN.Module):
    """Simplified decode head using SimNN operations"""
    
    def __init__(self, name: str, hidden_sizes: List[int], decoder_hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.name = name
        self.hidden_sizes = hidden_sizes
        self.decoder_hidden_size = decoder_hidden_size
        self.num_labels = num_labels
        
        # Linear layers for each encoder stage
        self.linear_c = []
        for i, hidden_size in enumerate(hidden_sizes):
            layer = SegformerMLPBlock(f"{name}.linear_c.{i}", hidden_size, decoder_hidden_size)
            self.linear_c.append(layer)
            setattr(self, f"linear_c_{i}", layer)
        
        # Fuse layer: takes concatenated features
        total_features = decoder_hidden_size * len(hidden_sizes)
        self.linear_fuse = SimNN.Linear(f"{name}.linear_fuse", total_features, decoder_hidden_size)
        
        # Final classifier
        self.classifier = SimNN.Linear(f"{name}.classifier", decoder_hidden_size, num_labels)
        
        # LayerNorm and activation
        self.layer_norm = F.LayerNorm(f"{name}.layer_norm", decoder_hidden_size)
        self.gelu = F.Gelu(f"{name}.gelu")
        
        super().link_op2module()
    
    def __call__(self, encoder_hidden_states: List[Any]) -> Any:
        # Process each encoder stage output and project to decoder_hidden_size
        projected_states = []
        for i, (hidden_state, mlp) in enumerate(zip(encoder_hidden_states, self.linear_c)):
            projected = mlp(hidden_state)
            projected_states.append(projected)
        
        # Concatenate along channel dimension (axis=2 for [batch, seq, channels])
        fused = T.cat(projected_states, dim=2)
        
        # Fuse layer
        fused = self.linear_fuse(fused)
        
        # LayerNorm + GELU
        fused = self.layer_norm(fused)
        fused = self.gelu(fused)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        count = 0
        for mlp in self.linear_c:
            count += mlp.analytical_param_count(lvl + 1)
        count += self.linear_fuse.analytical_param_count(lvl + 1)
        count += self.classifier.analytical_param_count(lvl + 1)
        return count


class SegformerEncoder(SimNN.Module):
    """Simplified encoder using SimNN operations"""
    
    def __init__(self, name: str, in_channels: int, hidden_sizes: List[int], image_size: int) -> None:
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.hidden_sizes = hidden_sizes
        self.image_size = image_size
        
        # Spatial dimensions at each stage
        self.spatial_dims = [image_size // 4, image_size // 8, image_size // 16, image_size // 32]
        self.seq_lens = [d * d for d in self.spatial_dims]
        
        # Patch embedding
        self.patch_embed = SimNN.Linear(f"{name}.patch_embed", in_channels, hidden_sizes[0])
        
        # Transformer stages
        self.stages = []
        self.layer_norms = []
        for i in range(len(hidden_sizes)):
            in_dim = hidden_sizes[i - 1] if i > 0 else hidden_sizes[0]
            out_dim = hidden_sizes[i]
            
            stage = SimNN.Linear(f"{name}.stage_{i}", in_dim, out_dim)
            self.stages.append(stage)
            setattr(self, f"stage_{i}", stage)
            
            ln = F.LayerNorm(f"{name}.ln_{i}", out_dim)
            self.layer_norms.append(ln)
            setattr(self, f"ln_{i}", ln)
        
        super().link_op2module()
    
    def __call__(self, pixel_values: Any) -> List[Any]:
        hidden_states = []
        x = pixel_values
        
        for i, (stage, ln) in enumerate(zip(self.stages, self.layer_norms)):
            if i == 0:
                x = self.patch_embed(x)
            
            x = stage(x)
            x = ln(x)
            
            hidden_states.append(x)
        
        return hidden_states
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        count = self.patch_embed.analytical_param_count(lvl + 1)
        for stage in self.stages:
            count += stage.analytical_param_count(lvl + 1)
        return count


class SegformerSemanticWorkload(SimNN.Module):
    """Complete Segformer workload using SimNN operations"""
    
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        
        self.bs = int(cfg.get('bs', 1))
        self.image_height = int(cfg.get('image_height', 512))
        self.image_width = int(cfg.get('image_width', 512))
        
        # SegFormer-B0 configuration
        self.hidden_sizes = [32, 64, 160, 256]
        self.decoder_hidden_size = 256
        self.num_labels = 150
        self.in_channels = 3
        
        # Sequence length after initial downsampling
        self.seq_len = (self.image_height // 4) * (self.image_width // 4)
        
        self.input_tensors: Dict[str, Any] = {}
        
        # Build encoder
        self.encoder = SegformerEncoder(
            f"{name}.encoder",
            self.in_channels,
            self.hidden_sizes,
            self.image_height
        )
        
        # Build decode head
        self.decode_head = SegformerDecodeHead(
            f"{name}.decode_head",
            self.hidden_sizes,
            self.decoder_hidden_size,
            self.num_labels
        )
        
        super().link_op2module()
    
    def create_input_tensors(self) -> None:
        """Create input tensors using F._from_shape (same pattern as BasicLLM)"""
        print("=== Starting Polaris Segformer Semantic Segmentation ===\n")
        
        self.input_tensors = {
            'pixel_values': F._from_shape(
                'pixel_values', 
                [self.bs, self.seq_len, self.in_channels], 
                is_param=False, 
                np_dtype=np.float32
            )
        }
        
        print(f"Input shape: {self.input_tensors['pixel_values'].shape}")
        print(f"Dimensions - N: {self.bs}, SeqLen: {self.seq_len}, C: {self.in_channels}")
        print()
        
        return
    
    def get_forward_graph(self) -> Any:
        GG = super()._get_forward_graph(self.input_tensors)
        return GG
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        count = 0
        count += self.encoder.analytical_param_count(lvl + 1)
        count += self.decode_head.analytical_param_count(lvl + 1)
        return count
    
    def __call__(self) -> Any:
        assert len(self.input_tensors) >= 1, "input_tensors missing! Call create_input_tensors() first."
        
        pixel_values = self.input_tensors['pixel_values']
        
        # Encoder
        hidden_states = self.encoder(pixel_values)
        
        print("=== Encoder Hidden States ===")
        for i, hs in enumerate(hidden_states):
            print(f"  Stage {i}: {hs.shape}")
        print()
        
        # Decode head
        logits = self.decode_head(hidden_states)
        
        print("=== Output ===")
        print(f"Final Logits shape: {logits.shape}")
        print()
        print("[PASSED] Segformer Semantic Segmentation completed!")
        
        return logits


def run_tests(name: str, cfg: Dict[str, Any]) -> SegformerSemanticWorkload:
    """Entry point for Polaris"""
    return SegformerSemanticWorkload(name=name, cfg=cfg)


if __name__ == "__main__":
    test_cfg = {'bs': 1, 'image_height': 512, 'image_width': 512}
    workload = run_tests("segformer_b0", test_cfg)
    workload.create_input_tensors()
    print(f"Parameter count: {workload.analytical_param_count()}")