# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import numpy as np
import traceback

# Force pathing
sys.path.insert(0, "/Users/suhanadas/suhana_polaris_fork")

import ttsim.front.functional.tensor_op as T
from workloads.segformer.tt.segformer_overlap_patch_embeddings import TtsimSegformerOverlapPatchEmbeddings

# --- THE PARAMETER LIST ---
# patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i
test_cases = [
    (7, 4, 3, 32, 1, 512, 512, 0),    # Stage 1: Down to 1/4 res (128x128)
    (3, 2, 32, 64, 1, 128, 128, 1),   # Stage 2: Down to 1/8 res (64x64)
    (3, 2, 64, 160, 1, 64, 64, 2),    # Stage 3: Down to 1/16 res (32x32)
    (3, 2, 160, 256, 1, 32, 32, 3),   # Stage 4: Down to 1/32 res (16x16)
]

def run_tests():
    print("=== Starting Polaris Segformer Overlap Patch Embeddings Verification ===")
    all_passed = True
    
    for case in test_cases:
        patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i = case
        
        test_name = f"Stage {patch_emb_i + 1} | In: {height}x{width}x{num_channels} | Out Channels: {hidden_size}"

        # 1. Mock Input Image/Feature Map [B, C, H, W]
        # (Note: In Polaris, standard convolution still expects NCHW format natively)
        pixel_values = T.SimTensor({
            "name": f"pixel_values_{patch_emb_i}",
            "data": np.random.randn(batch_size, num_channels, height, width).astype(np.float32),
            "shape": [batch_size, num_channels, height, width],
            "dtype": "float32"
        })

        # 2. Mock Parameters dynamically based on the current stage's sizes
        params = {
            "proj": {
                "weight": np.random.randn(hidden_size, num_channels, patch_size, patch_size).astype(np.float32),
                "bias": np.random.randn(hidden_size).astype(np.float32)
            },
            "layer_norm": {
                "weight": np.random.randn(hidden_size).astype(np.float32),
                "bias": np.random.randn(hidden_size).astype(np.float32)
            }
        }

        # 3. Initialize and Run
        try:
            model = TtsimSegformerOverlapPatchEmbeddings(
                name=f"test_patch_embed_{patch_emb_i}", 
                parameters=params, 
                stride=stride, 
                patch_size=patch_size
            )
            output, out_h, out_w = model(pixel_values)
            
            # 4. Verify Shape Math
            expected_h = height // stride
            expected_w = width // stride
            expected_seq_len = expected_h * expected_w
            expected_shape = [batch_size, expected_seq_len, hidden_size]
            
            if output.shape == expected_shape and out_h == expected_h and out_w == expected_w:
                print(f"[PASSED] {test_name} -> Output Shape: {output.shape}, H={out_h}, W={out_w}")
            else:
                print(f"[FAILED] {test_name} -> Expected {expected_shape}, got {output.shape}")
                all_passed = False
            
        except Exception as e:
            print(f"[ERROR]  {test_name}")
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("\n All Overlap Patch Embeddings configurations passed successfully!")

if __name__ == "__main__":
    run_tests()