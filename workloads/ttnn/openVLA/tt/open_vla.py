# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
OpenVLA.py - Polaris Version

A Vision-Language-Action model implementation for Tenstorrent hardware.
Entry point: test_openvla_model(name, device, cfg)

Architecture:
    - OpenVLAConfig: Model configuration
    - TTNNPrismaticProjector: Vision-to-LLM projection MLP
    - PerfCheckpoints: Performance timing utility
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# Path Setup
# ============================================================================

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

_project_root = os.path.abspath(os.path.join(_current_dir, '..', '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import ttsim.front.ttnn as ttnn

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

# Vision backbone configurations
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "siglip-vit-so400m": [224],
    "dinov2-vit-l": [224],
    "dinosiglip-vit-so-224px": [224, 224],
}

# LLM backbone configurations
LLM_BACKBONE_TO_HF_PATH: Dict[str, str] = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf",
}

# Valid backbone identifiers
VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH.keys())

# Vision encoder dimensions
DINOV2_EMBED_DIM = 1024
SIGLIP_EMBED_DIM = 1152
FUSED_VISION_DIM = DINOV2_EMBED_DIM + SIGLIP_EMBED_DIM  # 2176

# Default patch size for ViT models
PATCH_SIZE = 14

# ============================================================================
# TTNN Utility Functions
# ============================================================================


def ttnn_from_numpy(
    array: np.ndarray,
    dtype=None,
    layout=None,
    device=None,
) -> ttnn.Tensor:
    """
    Create TTNN tensor from numpy array.
    
    The order of operations matters for Polaris/ttsim compatibility:
        1. Tensor(numpy, device) - create tensor with device
        2. typecast              - change dtype (requires device)
        3. to_layout             - change memory layout
    
    Args:
        array: Input numpy array
        dtype: Target TTNN dtype (e.g., ttnn.bfloat16)
        layout: Target memory layout (e.g., ttnn.TILE_LAYOUT)
        device: TTNN device for tensor placement
    
    Returns:
        TTNN tensor with specified properties
    """
    # Ensure compatible dtype
    if array.dtype not in [np.float32, np.float16]:
        array = array.astype(np.float32)
    
    # Ensure contiguous memory layout
    array = np.ascontiguousarray(array)
    
    # Step 1: Create tensor on device
    if device is not None:
        tensor = ttnn.Tensor(array, device=device)
    else:
        tensor = ttnn.Tensor(array)
    
    # Step 2: Cast to target dtype
    if dtype is not None:
        tensor = ttnn.typecast(tensor, dtype)
    
    # Step 3: Convert to target layout
    if layout is not None:
        tensor = ttnn.to_layout(tensor, layout)
    
    return tensor


# ============================================================================
# Performance Monitoring
# ============================================================================


class PerfCheckpoints:
    """
    Performance checkpoint tracker for timing analysis.
    
    Usage:
        CHECKPOINTS.reset()
        CHECKPOINTS.checkpoint("start_operation")
        # ... do work ...
        CHECKPOINTS.checkpoint("end_operation")
        timings = CHECKPOINTS.analyze()
    """
    
    # Class-level storage for historical checkpoints
    checkpoints: ClassVar[Optional[List[Dict[str, float]]]] = None
    
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.present_keys_counter: Dict[str, int] = {}
        
        if PerfCheckpoints.checkpoints is None:
            PerfCheckpoints.checkpoints = []
    
    def checkpoint(self, key: str) -> None:
        """Record a timestamp for the given key."""
        if key not in self.present_keys_counter:
            self.present_keys_counter[key] = 0
        else:
            self.present_keys_counter[key] += 1
        
        indexed_key = f"{key}_{self.present_keys_counter[key]}"
        self.times[indexed_key] = time.time()
    
    def get_pairs(self) -> List[Tuple[str, str]]:
        """Find matching start/end checkpoint pairs."""
        pairs = []
        start_keys = [k for k in self.present_keys_counter if k.startswith("start")]
        
        for start_key in start_keys:
            end_key = start_key.replace("start", "end")
            if end_key in self.present_keys_counter:
                pairs.append((start_key, end_key))
        
        return pairs
    
    def analyze(self, pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, float]:
        """
        Calculate elapsed time for checkpoint pairs.
        
        Returns:
            Dictionary mapping "start->end" to elapsed seconds, sorted by duration
        """
        results = {}
        
        if pairs is None:
            pairs = self.get_pairs()
        
        for start_key, end_key in pairs:
            max_count = self.present_keys_counter.get(start_key, 0) + 1
            
            for counter in range(max_count):
                key1 = f"{start_key}_{counter}"
                key2 = f"{end_key}_{counter}"
                
                if key1 in self.times and key2 in self.times:
                    elapsed = self.times[key2] - self.times[key1]
                    results[f"{key1}->{key2}"] = elapsed
        
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    def reset(self) -> None:
        """Archive current checkpoints and reset for new measurement."""
        if self.times and PerfCheckpoints.checkpoints is not None:
            PerfCheckpoints.checkpoints.append(self.times)
        
        self.times = {}
        self.present_keys_counter = {}


# Global checkpoint instance
CHECKPOINTS = PerfCheckpoints()


# ============================================================================
# Model Configuration
# ============================================================================


@dataclass
class OpenVLAConfig:
    """
    Configuration for OpenVLA model.
    
    Attributes:
        model_type: Model identifier
        vision_backbone_id: Vision encoder type (e.g., "dinosiglip-vit-so-224px")
        llm_backbone_id: Language model type (e.g., "llama2-7b-pure")
        use_fused_vision_backbone: Whether to use DINOv2+SigLIP fusion
        hidden_size: LLM hidden dimension
        n_action_bins: Number of discrete action bins for output
    """
    
    model_type: str = "openvla"
    vision_backbone_id: str = "dinosiglip-vit-so-224px"
    llm_backbone_id: str = "llama2-7b-pure"
    use_fused_vision_backbone: Optional[bool] = None
    hidden_size: int = 4096
    n_action_bins: int = 256
    
    def __post_init__(self):
        """Validate configuration and set derived values."""
        # Validate backbone selections
        if self.vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(
                f"Invalid vision backbone: {self.vision_backbone_id}. "
                f"Valid options: {VALID_VISION_BACKBONES}"
            )
        
        if self.llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(
                f"Invalid LLM backbone: {self.llm_backbone_id}. "
                f"Valid options: {VALID_LLM_BACKBONES}"
            )
        
        # Auto-detect fused backbone usage
        if self.use_fused_vision_backbone is None:
            self.use_fused_vision_backbone = any(
                self.vision_backbone_id.startswith(prefix)
                for prefix in ["dinoclip", "dinosiglip"]
            )
    
    @property
    def vision_dim(self) -> int:
        """Get vision feature dimension based on backbone type."""
        if self.use_fused_vision_backbone:
            return FUSED_VISION_DIM
        return DINOV2_EMBED_DIM


# ============================================================================
# TTNN Projector Module
# ============================================================================


class TTNNPrismaticProjector:
    """
    TTNN-based MLP projector for vision-to-LLM feature projection.
    
    Architecture:
        Input (vision_dim) -> FC1 (4x expansion) -> GELU -> FC2 -> GELU -> FC3 -> Output (llm_dim)
    
    Args:
        use_fused_vision_backbone: Whether using fused DINOv2+SigLIP features
        vision_dim: Input feature dimension from vision encoder
        llm_dim: Output dimension matching LLM embedding size
        device: TTNN device for tensor operations
        weights: Optional pre-trained weights dictionary
    """
    
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        vision_dim: int,
        llm_dim: int,
        device,
        weights: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.device = device
        
        # Initialize weights
        if weights is not None:
            self._load_weights(weights)
        else:
            self._init_random_weights()
    
    def _load_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Load pre-trained weights from numpy arrays."""
        # FC1: vision_dim -> 4*vision_dim
        self.fc1_weight = ttnn_from_numpy(
            weights["fc1.weight"].T.astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.fc1_bias = ttnn_from_numpy(
            np.expand_dims(weights["fc1.bias"], 0).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        
        # FC2: 4*vision_dim -> llm_dim
        self.fc2_weight = ttnn_from_numpy(
            weights["fc2.weight"].T.astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.fc2_bias = ttnn_from_numpy(
            np.expand_dims(weights["fc2.bias"], 0).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        
        # FC3: llm_dim -> llm_dim
        self.fc3_weight = ttnn_from_numpy(
            weights["fc3.weight"].T.astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.fc3_bias = ttnn_from_numpy(
            np.expand_dims(weights["fc3.bias"], 0).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
    
    def _init_random_weights(self) -> None:
        """Initialize weights with random values (for testing)."""
        expansion_dim = 4 * self.vision_dim
        init_std = 0.02
        
        # FC1: vision_dim -> expansion_dim
        self.fc1_weight = ttnn_from_numpy(
            (np.random.randn(self.vision_dim, expansion_dim) * init_std).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.fc1_bias = ttnn_from_numpy(
            np.zeros((1, expansion_dim), dtype=np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        
        # FC2: expansion_dim -> llm_dim
        self.fc2_weight = ttnn_from_numpy(
            (np.random.randn(expansion_dim, self.llm_dim) * init_std).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.fc2_bias = ttnn_from_numpy(
            np.zeros((1, self.llm_dim), dtype=np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        
        # FC3: llm_dim -> llm_dim
        self.fc3_weight = ttnn_from_numpy(
            (np.random.randn(self.llm_dim, self.llm_dim) * init_std).astype(np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        self.fc3_bias = ttnn_from_numpy(
            np.zeros((1, self.llm_dim), dtype=np.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
    
    def forward(self, img_patches: ttnn.Tensor) -> ttnn.Tensor:
        """
        Project vision features to LLM embedding space.
        
        Args:
            img_patches: Vision features [batch, num_patches, vision_dim]
        
        Returns:
            Projected features [batch, num_patches, llm_dim]
        """
        # Layer 1: Linear + GELU
        x = ttnn.linear(
            img_patches,
            self.fc1_weight,
            bias=self.fc1_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            activation="gelu",
        )
        
        # Layer 2: Linear + GELU
        x = ttnn.linear(
            x,
            self.fc2_weight,
            bias=self.fc2_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            activation="gelu",
        )
        
        # Layer 3: Linear (no activation)
        x = ttnn.linear(
            x,
            self.fc3_weight,
            bias=self.fc3_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        
        return x
    
    def __call__(self, img_patches: ttnn.Tensor) -> ttnn.Tensor:
        """Enable calling instance as function."""
        return self.forward(img_patches)


# ============================================================================
# Polaris Workload Entry Point
# ============================================================================


def test_openvla_model(name: str, device, cfg: Dict) -> ttnn.Tensor:
    """
    OpenVLA model test - Polaris workload entry point.
    
    This function serves as the main interface for the Polaris testing framework.
    It creates a projector model and runs inference with synthetic vision features.
    
    Args:
        name: Workload name identifier (from Polaris)
        device: TTNN device handle
        cfg: Configuration dictionary with keys:
            - 'bs': Batch size (default: 1)
            - 'img_size': Input image size (default: 224)
            - 'in_chans': Number of input channels (default: 3)
    
    Returns:
        TTNN tensor output for Polaris graph capture
        Shape: [batch_size, num_patches, hidden_size]
    """

    print("\n" + "=" * 60)
    print(f"OpenVLA End-to-End Test - {name}")
    print("=" * 60)
    
    batch_size = cfg.get('bs', 1)
    img_size = cfg.get('img_size', 224)
    in_chans = cfg.get('in_chans', 3)
    
    print(f"Config: bs={batch_size}, img_size={img_size}, in_chans={in_chans}")
    
    config = OpenVLAConfig()
    vision_dim = config.vision_dim
    
    print(f"\nModel Configuration:")
    print(f"  Vision backbone: {'fused (DINOv2+SigLIP)' if config.use_fused_vision_backbone else 'DINOv2 only'}")
    print(f"  Vision dim: {vision_dim}")
    print(f"  LLM dim: {config.hidden_size}")
    
    print(f"\nCreating projector...")
    
    projector = TTNNPrismaticProjector(
        use_fused_vision_backbone=config.use_fused_vision_backbone or True,
        vision_dim=vision_dim,
        llm_dim=config.hidden_size,
        device=device,
        weights=None,  # Use random weights for testing
    )
    
    print("  Projector created successfully")
    
    num_patches = (img_size // PATCH_SIZE) ** 2  # 256 for 224x224
    
    print(f"\nCreating input tensor: [{batch_size}, {num_patches}, {vision_dim}]")
    
    # Generate synthetic vision features
    fake_vision_output = np.random.randn(
        batch_size, num_patches, vision_dim
    ).astype(np.float32)
    
    vision_features = ttnn_from_numpy(
        fake_vision_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    
    print("\nRunning projector forward pass...")
    
    CHECKPOINTS.reset()
    CHECKPOINTS.checkpoint("start_PROJECTORFORWARD")
    
    output = projector(vision_features)
    
    CHECKPOINTS.checkpoint("end_PROJECTORFORWARD")
    
    output_shape = list(output.shape)
    expected_shape = [batch_size, num_patches, config.hidden_size]
    
    print(f"\nResults:")
    print(f"  Output shape: {output_shape}")
    print(f"  Expected shape: {expected_shape}")
    
    if output_shape == expected_shape:
        print("  Status: PASSED ✓")
    else:
        print("  Status: FAILED ✗")
    
    timings = CHECKPOINTS.analyze()
    if timings:
        print(f"\nTimings:")
        print(json.dumps(timings, indent=4))
    
    print("=" * 60 + "\n")
    
    return output


def main():
    """
    Main entry point for standalone testing.
    
    Opens a TTNN device, runs the OpenVLA test, and reports results.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    
    print("=" * 60)
    print("OpenVLA Standalone Test")
    print("=" * 60)
    
    # Open device
    print("\nOpening TTNN device...")
    device = ttnn.open_device(device_id=0)
    print(f"  Device: {device}")
    
    try:
        # Run test with default configuration
        cfg = {
            'bs': 1,
            'img_size': 224,
            'in_chans': 3,
        }
        
        output = test_openvla_model(
            name="openvla_standalone_test",
            device=device,
            cfg=cfg,
        )
        
        print(f"\nTest completed successfully!")
        print(f"  Output shape: {list(output.shape)}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nClosing device...")
        ttnn.close_device(device)
        print("Done.")


if __name__ == "__main__":
    main()