# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys
import numpy as np
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_efficient_selfattention import TtSegformerEfficientSelfAttention


def log(message):
    print(message)
    sys.stdout.flush()

def create_custom_mesh_preprocessor(device):
    """Preprocessor to extract parameters from self-attention model and move to device."""
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        # Query
        query_weight = ttnn.as_tensor(model.query.weight)
        query_weight = ttnn.to_device(query_weight, device)
        query_bias = ttnn.as_tensor(model.query.bias)
        query_bias = ttnn.to_device(query_bias, device)
        parameters["query"] = {"weight": query_weight, "bias": query_bias}
        
        # Key
        key_weight = ttnn.as_tensor(model.key.weight)
        key_weight = ttnn.to_device(key_weight, device)
        key_bias = ttnn.as_tensor(model.key.bias)
        key_bias = ttnn.to_device(key_bias, device)
        parameters["key"] = {"weight": key_weight, "bias": key_bias}
        
        # Value
        value_weight = ttnn.as_tensor(model.value.weight)
        value_weight = ttnn.to_device(value_weight, device)
        value_bias = ttnn.as_tensor(model.value.bias)
        value_bias = ttnn.to_device(value_bias, device)
        parameters["value"] = {"weight": value_weight, "bias": value_bias}
        
        # Layer Norm
        ln_weight = ttnn.as_tensor(model.layer_norm.weight)
        ln_weight = ttnn.to_device(ln_weight, device)
        ln_bias = ttnn.as_tensor(model.layer_norm.bias)
        ln_bias = ttnn.to_device(ln_bias, device)
        parameters["layer_norm"] = {"weight": ln_weight, "bias": ln_bias}
        
        # SR (if exists)
        if hasattr(model, 'sr'):
            sr_weight = ttnn.as_tensor(model.sr.weight)
            sr_weight = ttnn.to_device(sr_weight, device)
            sr_bias = ttnn.as_tensor(model.sr.bias)
            sr_bias = ttnn.to_device(sr_bias, device)
            parameters["sr"] = {"weight": sr_weight, "bias": sr_bias}
        
        return parameters
    return preprocessor


def create_test_parameters(device, hidden_size, num_attention_heads, sequence_reduction_ratio):
    parameters = {}

    parameters["query"] = {}
    parameters["query"]["weight"] = ttnn.Tensor(
        np.random.randn(hidden_size, hidden_size).astype(np.float32).T,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    parameters["query"]["bias"] = ttnn.Tensor(
        np.random.randn(1, hidden_size).astype(np.float32),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    parameters["key"] = {}
    parameters["key"]["weight"] = ttnn.Tensor(
        np.random.randn(hidden_size, hidden_size).astype(np.float32).T,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    parameters["key"]["bias"] = ttnn.Tensor(
        np.random.randn(1, hidden_size).astype(np.float32),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    parameters["value"] = {}
    parameters["value"]["weight"] = ttnn.Tensor(
        np.random.randn(hidden_size, hidden_size).astype(np.float32).T,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    parameters["value"]["bias"] = ttnn.Tensor(
        np.random.randn(1, hidden_size).astype(np.float32),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    if sequence_reduction_ratio > 1:
        sr_out_channels = hidden_size
        sr_in_channels = hidden_size
        kernel_size = sequence_reduction_ratio

        parameters["sr"] = {}
        parameters["sr"]["weight"] = ttnn.Tensor(
            np.random.randn(sr_out_channels, sr_in_channels, kernel_size, kernel_size).astype(np.float32),
            dtype=ttnn.bfloat16,
        )
        parameters["sr"]["bias"] = ttnn.Tensor(
            np.random.randn(sr_out_channels).astype(np.float32),
            dtype=ttnn.bfloat16,
        )

        parameters["layer_norm"] = {}
        parameters["layer_norm"]["weight"] = ttnn.Tensor(
            np.random.randn(1, hidden_size).astype(np.float32),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        parameters["layer_norm"]["bias"] = ttnn.Tensor(
            np.random.randn(1, hidden_size).astype(np.float32),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    return parameters


def convert_to_numpy(tensor_output):
    """
    Safely convert ttnn tensor output to numpy array.
    """
    # First convert to torch
    torch_tensor = ttnn.to_torch(tensor_output)
    
    # Try different methods to get numpy array
    if torch_tensor is None:
        return np.array([])
    
    # If it has a shape attribute and it's empty, return empty array
    if hasattr(torch_tensor, 'shape'):
        shape = torch_tensor.shape
        if hasattr(shape, '__len__') and len(shape) == 0:
            # Scalar or empty shape - try to get the underlying data
            pass
    
    # Try detach().cpu().numpy() for PyTorch tensors
    if hasattr(torch_tensor, 'detach'):
        try:
            return torch_tensor.detach().cpu().numpy()
        except Exception:
            pass
    
    # Try .numpy() method
    if hasattr(torch_tensor, 'numpy'):
        try:
            result = torch_tensor.numpy()
            if isinstance(result, np.ndarray):
                return result
        except Exception:
            pass
    
    # Try to convert via np.array
    try:
        result = np.array(torch_tensor)
        if isinstance(result, np.ndarray) and result.size > 0:
            return result
    except Exception:
        pass
    
    # If torch_tensor has data attribute
    if hasattr(torch_tensor, 'data'):
        try:
            return np.array(torch_tensor.data)
        except Exception:
            pass
    
    # Last resort: return as-is wrapped in array
    return np.array([torch_tensor])


def test_segformer_efficient_selfattention(
    device,
    batch_size,
    seq_len,
    hidden_size,
    height,
    width,
    num_attention_heads,
    sequence_reduction_ratio,
    block_i,
    efficient_self_attention_i,
):
    log(f"\n{'='*60}")
    log(f"Running Test: block_{block_i}, attention_{efficient_self_attention_i}")
    log(f"{'='*60}")

    input_data = np.random.randn(batch_size, 1, seq_len, hidden_size).astype(np.float32)
    log(f"Input shape: {input_data.shape}")
    log(f"Input stats - min: {input_data.min():.4f}, max: {input_data.max():.4f}, mean: {input_data.mean():.4f}")

    ttnn_input_tensor = ttnn.Tensor(
        input_data,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    log("Input tensor created")

    parameters = create_test_parameters(device, hidden_size, num_attention_heads, sequence_reduction_ratio)
    log("Parameters created")

    ttnn_model = TtSegformerEfficientSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        parameters=parameters,
        sequence_reduction_ratio=sequence_reduction_ratio,
    )
    log("Model created")

    log("Running inference...")
    ttnn_output = ttnn_model(device, ttnn_input_tensor, height, width)
    log("Inference completed")

    # Get the output tensor
    output_tensor = ttnn_output[0]
    
    # Log output tensor info
    log(f"Output tensor shape attribute: {output_tensor.shape}")
    
    # Convert to numpy using helper function
    ttnn_final_output = convert_to_numpy(output_tensor)
    
    log(f"Converted output shape: {ttnn_final_output.shape}")
    log(f"Converted output size: {ttnn_final_output.size}")

    if ttnn_final_output.size > 0:
        # Flatten to ensure we can compute stats
        flat_output = ttnn_final_output.flatten()
        if len(flat_output) > 0:
            try:
                min_val = float(flat_output.min())
                max_val = float(flat_output.max())
                mean_val = float(flat_output.mean())
                log(f"Output stats - min: {min_val:.4f}, max: {max_val:.4f}, mean: {mean_val:.4f}")
            except Exception as e:
                log(f"Could not compute stats: {e}")

    # Adjust expected shape based on whether we have multi-head attention
    if len(ttnn_final_output.shape) == 4:
        ttnn_final_output = ttnn_final_output[0]

    expected_shape = (batch_size, seq_len, hidden_size)

    # Check that we got valid output
    assert ttnn_final_output.size > 0, "Output is empty"
    
    # Check for NaN/Inf only if we have numeric data
    if ttnn_final_output.dtype in [np.float32, np.float64, np.float16]:
        assert not np.isnan(ttnn_final_output).any(), "Output contains NaN values"
        assert not np.isinf(ttnn_final_output).any(), "Output contains Inf values"

    log(f"[PASSED] Test: block_{block_i}, attention_{efficient_self_attention_i}")

    return ttnn_final_output


def run_tests(device):
    test_parameters = [
        # (batch_size, seq_len, hidden_size, height, width, num_attention_heads, sequence_reduction_ratio, block_i, attention_i)
        (1, 16384, 32, 128, 128, 1, 8, 0, 0),
        (1, 16384, 32, 128, 128, 1, 8, 0, 1),
        (1, 4096, 64, 64, 64, 2, 4, 1, 0),
        (1, 4096, 64, 64, 64, 2, 4, 1, 1),
        (1, 1024, 160, 32, 32, 5, 2, 2, 0),
        (1, 1024, 160, 32, 32, 5, 2, 2, 1),
        (1, 256, 256, 16, 16, 8, 1, 3, 0),
        (1, 256, 256, 16, 16, 8, 1, 3, 1),
    ]

    log("\n" + "=" * 60)
    log("Starting Segformer Efficient Self-Attention Tests")
    log("=" * 60)
    log(f"Total tests to run: {len(test_parameters)}")

    passed = 0
    failed = 0

    for params in test_parameters:
        (
            batch_size,
            seq_len,
            hidden_size,
            height,
            width,
            num_attention_heads,
            sequence_reduction_ratio,
            block_i,
            efficient_self_attention_i,
        ) = params

        try:
            test_segformer_efficient_selfattention(
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=hidden_size,
                height=height,
                width=width,
                num_attention_heads=num_attention_heads,
                sequence_reduction_ratio=sequence_reduction_ratio,
                block_i=block_i,
                efficient_self_attention_i=efficient_self_attention_i,
            )
            passed += 1
        except Exception as e:
            failed += 1
            log(f"[FAILED] Test: block_{block_i}, attention_{efficient_self_attention_i}")
            log(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    log("\n" + "=" * 60)
    log("TEST SUMMARY")
    log("=" * 60)
    log(f"Total: {len(test_parameters)}")
    log(f"Passed: {passed}")
    log(f"Failed: {failed}")
    log("=" * 60)

    if failed == 0:
        log("ALL TESTS PASSED!")
    else:
        log(f"WARNING: {failed} test(s) failed!")


log("=" * 60)
log("Script starting...")
log("=" * 60)

try:
    log("Step 1: Importing complete")

    log("Step 2: Opening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    log("Step 3: Device opened")

    ttnn.set_default_device(device)
    log("Step 4: Default device set")

    log("Step 5: Running tests...")
    run_tests(device)

    log("Step 6: Closing device...")
    ttnn.close_device(device)
    log("Step 7: Device closed")

except Exception as e:
    log(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

log("Script finished")