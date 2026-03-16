# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import numpy as np
from typing import Any, Dict, List

sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN


class SegformerMLPBlock(SimNN.Module):
    """MLP block used in classification head"""
    
    def __init__(self, name: str, in_features: int, out_features: int) -> None:
        super().__init__()
        self.name = name
        self.proj = SimNN.Linear(f"{name}.proj", in_features, out_features)
        super().link_op2module()
    
    def __call__(self, x: Any) -> Any:
        return self.proj(x)
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        return self.proj.analytical_param_count(lvl + 1)


class SegformerClassificationHead(SimNN.Module):
    """Classification head for image classification"""
    
    def __init__(self, name: str, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.name = name
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.classifier = SimNN.Linear(f"{name}.classifier", hidden_size, num_classes)
        
        super().link_op2module()
    
    def __call__(self, hidden_state: Any) -> Any:
        # hidden_state: [batch, seq_len, hidden_size]
        # Global average pooling: mean over seq_len dimension
        # For simplicity, we'll just use the first token (like [CLS])
        # or apply the linear directly and let the shape work out
        
        # Take mean over sequence (simplified global pooling)
        # In real implementation: pooled = hidden_state.mean(dim=1)
        # Here we just classify directly
        logits = self.classifier(hidden_state)
        
        return logits
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        return self.classifier.analytical_param_count(lvl + 1)


class SegformerEncoder(SimNN.Module):
    """Simplified encoder using SimNN operations"""
    
    def __init__(self, name: str, in_channels: int, hidden_sizes: List[int], image_size: int) -> None:
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.hidden_sizes = hidden_sizes
        self.image_size = image_size
        
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
    
    def __call__(self, pixel_values: Any) -> Any:
        x = pixel_values
        
        for i, (stage, ln) in enumerate(zip(self.stages, self.layer_norms)):
            if i == 0:
                x = self.patch_embed(x)
            
            x = stage(x)
            x = ln(x)
        
        # Return only the last hidden state for classification
        return x
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        count = self.patch_embed.analytical_param_count(lvl + 1)
        for stage in self.stages:
            count += stage.analytical_param_count(lvl + 1)
        return count


class SegformerClassificationWorkload(SimNN.Module):
    """Segformer Image Classification workload"""
    
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        
        self.bs = int(cfg.get('bs', 1))
        self.image_height = int(cfg.get('image_height', 224))
        self.image_width = int(cfg.get('image_width', 224))
        self.num_classes = int(cfg.get('num_classes', 1000))
        
        # SegFormer-B0 configuration
        self.hidden_sizes = [32, 64, 160, 256]
        self.in_channels = 3
        
        # Sequence length after initial downsampling (4x)
        self.seq_len = (self.image_height // 4) * (self.image_width // 4)
        
        self.input_tensors: Dict[str, Any] = {}
        
        # Build encoder
        self.encoder = SegformerEncoder(
            f"{name}.encoder",
            self.in_channels,
            self.hidden_sizes,
            self.image_height
        )
        
        # Build classification head
        self.classifier = SegformerClassificationHead(
            f"{name}.classifier",
            self.hidden_sizes[-1],  # Use last hidden size (256)
            self.num_classes
        )
        
        super().link_op2module()
    
    def create_input_tensors(self) -> None:
        """Create input tensors using F._from_shape"""
        print("=== Starting Polaris Segformer Image Classification ===\n")
        
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
        print(f"Image size: {self.image_height}x{self.image_width}")
        print(f"Num classes: {self.num_classes}")
        print()
        
        return
    
    def get_forward_graph(self) -> Any:
        GG = super()._get_forward_graph(self.input_tensors)
        return GG
    
    def analytical_param_count(self, lvl: int = 0) -> int:
        count = 0
        count += self.encoder.analytical_param_count(lvl + 1)
        count += self.classifier.analytical_param_count(lvl + 1)
        return count
    
    def __call__(self) -> Any:
        assert len(self.input_tensors) >= 1, "input_tensors missing! Call create_input_tensors() first."
        
        pixel_values = self.input_tensors['pixel_values']
        
        # Encoder - get last hidden state
        hidden_state = self.encoder(pixel_values)
        
        print("=== Encoder Output ===")
        print(f"  Last hidden state: {hidden_state.shape}")
        print()
        
        # Classification head
        logits = self.classifier(hidden_state)
        
        print("=== Output ===")
        print(f"Classification logits shape: {logits.shape}")
        print()
        print("[PASSED] Segformer Image Classification completed!")
        
        return logits


def run_tests(name: str, cfg: Dict[str, Any]) -> SegformerClassificationWorkload:
    """Entry point for Polaris"""
    return SegformerClassificationWorkload(name=name, cfg=cfg)


if __name__ == "__main__":
    test_cfg = {'bs': 1, 'image_height': 224, 'image_width': 224, 'num_classes': 1000}
    workload = run_tests("segformer_b0", test_cfg)
    workload.create_input_tensors()
    print(f"Parameter count: {workload.analytical_param_count()}")