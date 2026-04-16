# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_layer import TtSegformerLayer


# --- Mock Classes for Encoder Test ---
class MockLinear:
    """Simulates torch.nn.Linear"""
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features).astype(np.float32)
        self.bias = np.random.randn(out_features).astype(np.float32)


class MockConv2d:
    """Simulates torch.nn.Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        self.weight = np.random.randn(out_channels, in_channels // groups, kernel_size, kernel_size).astype(np.float32)
        self.bias = np.random.randn(out_channels).astype(np.float32)


class MockLayerNorm:
    """Simulates torch.nn.LayerNorm"""
    def __init__(self, hidden_size):
        self.weight = np.random.randn(hidden_size).astype(np.float32)
        self.bias = np.random.randn(hidden_size).astype(np.float32)


class MockSegformerEfficientSelfAttention:
    """Mock for efficient self attention"""
    def __init__(self, hidden_size, num_attention_heads, sequence_reduction_ratio):
        self.query = MockLinear(hidden_size, hidden_size)
        self.key = MockLinear(hidden_size, hidden_size)
        self.value = MockLinear(hidden_size, hidden_size)
        
        if sequence_reduction_ratio > 1:
            self.sr = MockConv2d(hidden_size, hidden_size, sequence_reduction_ratio, sequence_reduction_ratio, 0)
            self.layer_norm = MockLayerNorm(hidden_size)
        else:
            self.sr = None
            self.layer_norm = None


class MockSegformerSelfOutput:
    """Mock for self attention output projection"""
    def __init__(self, hidden_size):
        self.dense = MockLinear(hidden_size, hidden_size)


class MockSegformerAttention:
    """Mock for complete attention module"""
    def __init__(self, hidden_size, num_attention_heads, sequence_reduction_ratio):
        self.self_attention = MockSegformerEfficientSelfAttention(
            hidden_size, num_attention_heads, sequence_reduction_ratio
        )
        self.output = MockSegformerSelfOutput(hidden_size)


class MockSegformerMLP:
    """Mock for MLP module"""
    def __init__(self, hidden_size, mlp_ratio):
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.dense1 = MockLinear(hidden_size, mlp_hidden_size)
        self.dwconv = MockConv2d(mlp_hidden_size, mlp_hidden_size, 3, 1, 1, groups=mlp_hidden_size)
        self.dense2 = MockLinear(mlp_hidden_size, hidden_size)


class MockSegformerLayer:
    """Mock for complete Segformer layer"""
    def __init__(self, hidden_size, num_attention_heads, sr_ratio):
        self.layer_norm_1 = MockLayerNorm(hidden_size)
        self.attention = MockSegformerAttention(hidden_size, num_attention_heads, sr_ratio)
        self.layer_norm_2 = MockLayerNorm(hidden_size)
        self.mlp = MockSegformerMLP(hidden_size, mlp_ratio=4)


def create_custom_mesh_preprocessor(device):
    """
    Preprocessor to extract parameters from mock Segformer layer and move to device.
    Used by the encoder test to create layer parameters.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        if isinstance(model, MockSegformerLayer):
            # LayerNorm 1
            parameters["layer_norm_1"] = {
                "weight": ttnn.to_device(ttnn.as_tensor(model.layer_norm_1.weight), device),
                "bias": ttnn.to_device(ttnn.as_tensor(model.layer_norm_1.bias), device),
            }
            
            # Attention
            parameters["attention"] = {
                "self": {
                    "query": {
                        "weight": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.query.weight.T), device),
                        "bias": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.query.bias), device),
                    },
                    "key": {
                        "weight": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.key.weight.T), device),
                        "bias": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.key.bias), device),
                    },
                    "value": {
                        "weight": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.value.weight.T), device),
                        "bias": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.value.bias), device),
                    },
                },
                "output": {
                    "dense": {
                        "weight": ttnn.to_device(ttnn.as_tensor(model.attention.output.dense.weight.T), device),
                        "bias": ttnn.to_device(ttnn.as_tensor(model.attention.output.dense.bias), device),
                    },
                },
            }
            
            # SR and layer norm if exists
            if model.attention.self_attention.sr is not None:
                parameters["attention"]["self"]["sr"] = {
                    "weight": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.sr.weight), device),
                    "bias": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.sr.bias), device),
                }
                parameters["attention"]["self"]["layer_norm"] = {
                    "weight": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.layer_norm.weight), device),
                    "bias": ttnn.to_device(ttnn.as_tensor(model.attention.self_attention.layer_norm.bias), device),
                }
            else:
                parameters["attention"]["self"]["sr"] = None
                parameters["attention"]["self"]["layer_norm"] = None
            
            # LayerNorm 2
            parameters["layer_norm_2"] = {
                "weight": ttnn.to_device(ttnn.as_tensor(model.layer_norm_2.weight), device),
                "bias": ttnn.to_device(ttnn.as_tensor(model.layer_norm_2.bias), device),
            }
            
            # MLP
            # MockLinear creates weight as [out_features, in_features] (PyTorch convention)
            # TtSegformerMixFFN expects [out_features, in_features] and does transpose internally
            # So we should NOT transpose here
            parameters["mlp"] = {
                "dense1": {
                    "weight": ttnn.to_device(ttnn.as_tensor(model.mlp.dense1.weight), device),  # NO .T
                    "bias": ttnn.to_device(ttnn.as_tensor(model.mlp.dense1.bias), device),
                },
                "dwconv": {
                    "weight": model.mlp.dwconv.weight,
                    "bias": model.mlp.dwconv.bias,
                },
                "dense2": {
                    "weight": ttnn.to_device(ttnn.as_tensor(model.mlp.dense2.weight), device),  # NO .T
                    "bias": ttnn.to_device(ttnn.as_tensor(model.mlp.dense2.bias), device),
                },
            }
        
        return parameters
    
    return preprocessor

def create_test_parameters(device, hidden_size, num_attention_heads, sequence_reduction_ratio, mlp_ratio):
    """Create test parameters for SegformerLayer"""
    
    # LayerNorm 1
    layer_norm_1_weight = np.random.randn(hidden_size).astype(np.float32)
    layer_norm_1_bias = np.random.randn(hidden_size).astype(np.float32)
    
    # Attention parameters
    attention_params = create_attention_parameters(device, hidden_size, num_attention_heads, sequence_reduction_ratio)
    
    # MLP parameters
    mlp_hidden_size = int(hidden_size * mlp_ratio)
    mlp_params = create_mlp_parameters(device, hidden_size, mlp_hidden_size)
    
    # LayerNorm 2
    layer_norm_2_weight = np.random.randn(hidden_size).astype(np.float32)
    layer_norm_2_bias = np.random.randn(hidden_size).astype(np.float32)
    
    parameters = {
        "layer_norm_1": {
            "weight": ttnn.to_device(ttnn.as_tensor(layer_norm_1_weight), device),
            "bias": ttnn.to_device(ttnn.as_tensor(layer_norm_1_bias), device),
        },
        "attention": attention_params,
        "mlp": mlp_params,
        "layer_norm_2": {
            "weight": ttnn.to_device(ttnn.as_tensor(layer_norm_2_weight), device),
            "bias": ttnn.to_device(ttnn.as_tensor(layer_norm_2_bias), device),
        },
    }
    
    return parameters


def create_attention_parameters(device, hidden_size, num_attention_heads, sequence_reduction_ratio):
    """Create attention submodule parameters"""
    # Query
    query_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    query_bias = np.random.randn(hidden_size).astype(np.float32)
    
    # Key
    key_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    key_bias = np.random.randn(hidden_size).astype(np.float32)
    
    # Value
    value_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    value_bias = np.random.randn(hidden_size).astype(np.float32)
    
    # SR (sequence reduction) - if ratio > 1
    if sequence_reduction_ratio > 1:
        sr_weight = np.random.randn(hidden_size, hidden_size, sequence_reduction_ratio, sequence_reduction_ratio).astype(np.float32)
        sr_bias = np.random.randn(hidden_size).astype(np.float32)
        layer_norm_weight = np.random.randn(hidden_size).astype(np.float32)
        layer_norm_bias = np.random.randn(hidden_size).astype(np.float32)
        
        sr_params = {
            "weight": ttnn.to_device(ttnn.as_tensor(sr_weight), device),
            "bias": ttnn.to_device(ttnn.as_tensor(sr_bias), device),
        }
        norm_params = {
            "weight": ttnn.to_device(ttnn.as_tensor(layer_norm_weight), device),
            "bias": ttnn.to_device(ttnn.as_tensor(layer_norm_bias), device),
        }
    else:
        sr_params = None
        norm_params = None
    
    # Projection (for output)
    proj_weight = np.random.randn(hidden_size, hidden_size).astype(np.float32)
    proj_bias = np.random.randn(hidden_size).astype(np.float32)
    
    # Return structure matching TtSegformerAttention's expectations
    return {
        "self": {  # Parameters for TtSegformerEfficientSelfAttention
            "query": {
                "weight": ttnn.to_device(ttnn.as_tensor(query_weight.T), device),
                "bias": ttnn.to_device(ttnn.as_tensor(query_bias), device),
            },
            "key": {
                "weight": ttnn.to_device(ttnn.as_tensor(key_weight.T), device),
                "bias": ttnn.to_device(ttnn.as_tensor(key_bias), device),
            },
            "value": {
                "weight": ttnn.to_device(ttnn.as_tensor(value_weight.T), device),
                "bias": ttnn.to_device(ttnn.as_tensor(value_bias), device),
            },
            "sr": sr_params,
            "layer_norm": norm_params,
        },
        "output": {  # Parameters for TtSegformerSelfOutput
            "dense": {
                "weight": ttnn.to_device(ttnn.as_tensor(proj_weight.T), device),
                "bias": ttnn.to_device(ttnn.as_tensor(proj_bias), device),
            },
        },
    }


def create_mlp_parameters(device, in_features, hidden_features):
    """Create MLP submodule parameters"""
    out_features = in_features
    
    dense1_weight = np.random.randn(in_features, hidden_features).astype(np.float32)
    dense1_bias = np.random.randn(hidden_features).astype(np.float32)
    
    dwconv_weight = np.random.randn(hidden_features, 1, 3, 3).astype(np.float32)
    dwconv_bias = np.random.randn(hidden_features).astype(np.float32)
    
    dense2_weight = np.random.randn(hidden_features, out_features).astype(np.float32)
    dense2_bias = np.random.randn(out_features).astype(np.float32)
    
    return {
        "dense1": {
            "weight": ttnn.to_device(ttnn.as_tensor(dense1_weight.T), device),
            "bias": ttnn.to_device(ttnn.as_tensor(dense1_bias), device),
        },
        "dwconv": {
            "weight": dwconv_weight,
            "bias": dwconv_bias,
        },
        "dense2": {
            "weight": ttnn.to_device(ttnn.as_tensor(dense2_weight.T), device),
            "bias": ttnn.to_device(ttnn.as_tensor(dense2_bias), device),
        },
    }


def test_segformer_layer(
    device,
    batch_size,
    seq_len,
    hidden_size,
    height,
    width,
    num_attention_heads,
    sequence_reduction_ratio,
    mlp_ratio,
    block_i,
    segformer_i,
):
    """Test SegformerLayer module"""
    print(f"\n[TEST] Block {block_i} | Layer {segformer_i}")
    print(f"       hidden={hidden_size}, heads={num_attention_heads}, sr_ratio={sequence_reduction_ratio}")
    print(f"       Input: [{batch_size}, {seq_len}, {hidden_size}], H={height}, W={width}")
    
    try:
        # Create input
        np.random.seed(42)
        input_data = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        
        # Convert to ttnn tensor
        ttnn_input = ttnn.as_tensor(input_data)
        ttnn_input = ttnn.to_device(ttnn_input, device)
        
        # Create parameters
        parameters = create_test_parameters(
            device, hidden_size, num_attention_heads, sequence_reduction_ratio, mlp_ratio
        )
        
        # Create model
        model = TtSegformerLayer(
            name=f"layer_b{block_i}_i{segformer_i}",
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            parameters=parameters,
            mlp_ratio=mlp_ratio,
        )
        
        # Run model
        ttnn_output = model(
            device,
            ttnn_input,
            height=height,
            width=width,
            output_attentions=False,
        )
        
        # Get output (first element of tuple)
        layer_output = ttnn_output[0]
        
        # Get output shape
        if hasattr(layer_output, 'shape'):
            actual_shape = tuple(layer_output.shape)
        else:
            actual_shape = tuple(layer_output.data.shape)
        
        # Verify shape - output should be [batch, seq_len, hidden_size]
        expected_shape = (batch_size, seq_len, hidden_size)
        
        if actual_shape == expected_shape:
            print(f"[PASS] Block {block_i} | Layer {segformer_i} | Output shape: {actual_shape}")
            return True
        else:
            print(f"[FAIL] Block {block_i} | Layer {segformer_i}")
            print(f"       Expected shape: {expected_shape}, Got: {actual_shape}")
            return False
        
    except Exception as e:
        print(f"[ERROR] Block {block_i} | Layer {segformer_i} -> {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(device):
    """Run all SegformerLayer tests"""
    print("\n" + "="*80)
    print("=== Polaris Segformer Layer Tests ===")
    print("="*80)
    
    test_configs = [
        # (batch, seq_len, hidden, height, width, heads, sr_ratio, mlp_ratio, block_i, layer_i)
        (1, 16384, 32, 128, 128, 1, 8, 4, 0, 0),
        (1, 16384, 32, 128, 128, 1, 8, 4, 0, 1),
        (1, 4096, 64, 64, 64, 2, 4, 4, 1, 0),
        (1, 4096, 64, 64, 64, 2, 4, 4, 1, 1),
        (1, 1024, 160, 32, 32, 5, 2, 4, 2, 0),
        (1, 1024, 160, 32, 32, 5, 2, 4, 2, 1),
        (1, 256, 256, 16, 16, 8, 1, 4, 3, 0),
        (1, 256, 256, 16, 16, 8, 1, 4, 3, 1),
    ]
    
    passed_count = 0
    
    for config in test_configs:
        (batch_size, seq_len, hidden_size, height, width, 
         num_attention_heads, sequence_reduction_ratio, mlp_ratio, block_i, segformer_i) = config
        
        if test_segformer_layer(
            device, batch_size, seq_len, hidden_size, height, width,
            num_attention_heads, sequence_reduction_ratio, mlp_ratio, block_i, segformer_i
        ):
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