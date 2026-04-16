# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_dwconv import TtSegformerDWConv

def create_test_parameters(dim):
    """Create test parameters for DWConv"""
    parameters = {
        "dwconv": {
            "weight": np.random.randn(dim, 1, 3, 3).astype(np.float32) * 0.02,
            "bias": np.random.randn(dim).astype(np.float32) * 0.01
        }
    }
    return parameters

def test_segformer_dwconv(device, batch_size, seq_len, dim, height, width, block_idx, dwconv_idx):
    """Test DWConv module - just verify it runs and shapes are correct"""
    print(f"\n[TEST] Block {block_idx} | DWConv {dwconv_idx} | Seq {seq_len} | Dim {dim}")
    
    try:
        # Create input
        np.random.seed(42)
        input_data = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        
        # Create parameters
        parameters = create_test_parameters(dim)
        
        # Convert input to ttnn tensor
        ttnn_input = ttnn.as_tensor(input_data)
        ttnn_input = ttnn.to_device(ttnn_input, device)
        
        # TT model
        model = TtSegformerDWConv(
            name=f"dwconv_b{block_idx}_i{dwconv_idx}",
            parameters=parameters,
            dim=dim
        )
        
        # Get TT output
        ttnn_output, out_h, out_w = model(device, ttnn_input, height, width)
        
        # Get output shape
        if hasattr(ttnn_output, 'shape'):
            actual_shape = tuple(ttnn_output.shape)
        else:
            actual_shape = tuple(ttnn_output.data.shape)
        
        # Verify shape - output should be (batch, height, width, dim)
        expected_shape = (batch_size, height, width, dim)
        
        if actual_shape == expected_shape and out_h == height and out_w == width:
            print(f"[PASS] Block {block_idx} | DWConv {dwconv_idx} | Shape: {actual_shape}")
            return True
        else:
            print(f"[FAIL] Block {block_idx} | DWConv {dwconv_idx}")
            print(f"       Expected shape: {expected_shape}, Got: {actual_shape}")
            print(f"       Expected H,W: ({height},{width}), Got: ({out_h},{out_w})")
            return False
        
    except Exception as e:
        print(f"[ERROR] Block {block_idx} | DWConv {dwconv_idx} -> {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests(device):
    """Run all DWConv tests"""
    print("\n" + "="*60)
    print("=== Polaris Segformer DWConv Tests ===")
    print("NOTE: Using placeholder implementation (ttsim limitation)")
    print("="*60)
    
    test_configs = [
        # (batch, seq_len, dim, height, width, block_idx, dwconv_idx)
        (1, 16384, 128, 128, 128, 0, 0),
        (1, 16384, 128, 128, 128, 0, 1),
        (1, 4096, 256, 64, 64, 1, 0),
        (1, 4096, 256, 64, 64, 1, 1),
        (1, 1024, 640, 32, 32, 2, 0),
        (1, 1024, 640, 32, 32, 2, 1),
        (1, 256, 1024, 16, 16, 3, 0),
        (1, 256, 1024, 16, 16, 3, 1),
    ]
    
    passed_count = 0
    
    for config in test_configs:
        batch, seq_len, dim, height, width, block_idx, dwconv_idx = config
        if test_segformer_dwconv(device, batch, seq_len, dim, height, width, block_idx, dwconv_idx):
            passed_count += 1
    
    # Print summary
    print("\n" + "="*60)
    print(f"Results: {passed_count}/{len(test_configs)} passed")
    print("="*60)
    
    return passed_count == len(test_configs)

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    print(f"Device opened: {device}")
    
    try:
        all_passed = run_all_tests(device)
        sys.exit(0 if all_passed else 1)
    finally:
        ttnn.close_device(device)
        print("Device closed.")