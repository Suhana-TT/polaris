# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_mix_ffn import TtSegformerMixFFN

def create_test_parameters(device, in_features, hidden_features, out_features):
    """Create test parameters for MixFFN and convert to ttnn tensors"""
    # Create numpy parameters
    # ttnn.as_tensor transposes, so we create transposed versions
    dense1_weight = np.random.randn(in_features, hidden_features).astype(np.float32) * 0.02
    dense1_bias = np.random.randn(hidden_features).astype(np.float32) * 0.01
    
    dwconv_weight = np.random.randn(hidden_features, 1, 3, 3).astype(np.float32) * 0.02
    dwconv_bias = np.random.randn(hidden_features).astype(np.float32) * 0.01
    
    dense2_weight = np.random.randn(hidden_features, out_features).astype(np.float32) * 0.02
    dense2_bias = np.random.randn(out_features).astype(np.float32) * 0.01
    
    # Convert to ttnn tensors - ttnn.as_tensor will transpose them
    # So [in, out] becomes [out, in] which is what matmul expects
    parameters = {
        "dense1": {
            "weight": ttnn.to_device(ttnn.as_tensor(dense1_weight.T), device),  # Transpose!
            "bias": ttnn.to_device(ttnn.as_tensor(dense1_bias), device)
        },
        "dwconv": {
            "weight": dwconv_weight,  # Keep as numpy for dwconv (it doesn't use it)
            "bias": dwconv_bias
        },
        "dense2": {
            "weight": ttnn.to_device(ttnn.as_tensor(dense2_weight.T), device),  # Transpose!
            "bias": ttnn.to_device(ttnn.as_tensor(dense2_bias), device)
        }
    }
    return parameters

def test_segformer_mix_ffn(device, in_features, hidden_features, out_features, 
                           batch_size, seq_len, height, width, block_i, mixffn_i):
    """Test MixFFN module - verify it runs and shapes are correct"""
    print(f"\n[TEST] Block {block_i} | MixFFN {mixffn_i}")
    print(f"       in={in_features}, hidden={hidden_features}, out={out_features}")
    print(f"       Input: [{batch_size}, {seq_len}, {in_features}], H={height}, W={width}")
    
    try:
        # Create input
        np.random.seed(42)
        input_data = np.random.randn(batch_size, seq_len, in_features).astype(np.float32)
        
        # Convert to ttnn tensor
        ttnn_input = ttnn.as_tensor(input_data)
        ttnn_input = ttnn.to_device(ttnn_input, device)
        
        # Create parameters (already converted to ttnn tensors)
        parameters = create_test_parameters(device, in_features, hidden_features, out_features)
        
        # Create model
        model = TtSegformerMixFFN(
            name=f"mixffn_b{block_i}_i{mixffn_i}",
            parameters=parameters,
            config=None,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
        )
        
        # Run model
        ttnn_output = model(device, ttnn_input, height=height, width=width)
        
        # Get output shape
        if hasattr(ttnn_output, 'shape'):
            actual_shape = tuple(ttnn_output.shape)
        else:
            actual_shape = tuple(ttnn_output.data.shape)
        
        # Verify shape - output should be [batch, seq_len, out_features]
        expected_shape = (batch_size, seq_len, out_features)
        
        if actual_shape == expected_shape:
            print(f"[PASS] Block {block_i} | MixFFN {mixffn_i} | Output shape: {actual_shape}")
            return True
        else:
            print(f"[FAIL] Block {block_i} | MixFFN {mixffn_i}")
            print(f"       Expected shape: {expected_shape}, Got: {actual_shape}")
            return False
        
    except Exception as e:
        print(f"[ERROR] Block {block_i} | MixFFN {mixffn_i} -> {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests(device):
    """Run all MixFFN tests"""
    print("\n" + "="*80)
    print("=== Polaris Segformer MixFFN Tests ===")
    print("="*80)
    
    test_configs = [
        # (in_features, hidden_features, out_features, batch, seq_len, height, width, block_i, mixffn_i)
        (32, 128, 32, 1, 16384, 128, 128, 0, 0),
        (32, 128, 32, 1, 16384, 128, 128, 0, 1),
        (64, 256, 64, 1, 4096, 64, 64, 1, 0),
        (64, 256, 64, 1, 4096, 64, 64, 1, 1),
        (160, 640, 160, 1, 1024, 32, 32, 2, 0),
        (160, 640, 160, 1, 1024, 32, 32, 2, 1),
        (256, 1024, 256, 1, 256, 16, 16, 3, 0),
        (256, 1024, 256, 1, 256, 16, 16, 3, 1),
    ]
    
    passed_count = 0
    
    for config in test_configs:
        (in_features, hidden_features, out_features, batch_size, 
         seq_len, height, width, block_i, mixffn_i) = config
        
        if test_segformer_mix_ffn(device, in_features, hidden_features, out_features,
                                 batch_size, seq_len, height, width, block_i, mixffn_i):
            passed_count += 1
    
    # Print summary
    print("\n" + "="*80)
    print(f"Results: {passed_count}/{len(test_configs)} passed")
    print("="*80)
    
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