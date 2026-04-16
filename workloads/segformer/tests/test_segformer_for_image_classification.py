# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import math
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_for_image_classification import (
    TtSegformerForImageClassification,
    TtSegFormerImageClassifierOutput,
)
from workloads.segformer.tests.test_segformer_model import (
    create_custom_mesh_preprocessor as custom_preprocessor_main_model,
    MockSegformerModel,
)


def create_polaris_tensor(numpy_array, device):
    """Convert numpy array to Polaris ttnn.Tensor and move to device."""
    tensor = ttnn.as_tensor(numpy_array.astype(np.float32))
    tensor = ttnn.to_device(tensor, device)
    return tensor


def tensor_to_numpy(tensor):
    """Safely convert ttnn tensor to numpy array."""
    try:
        result = ttnn.to_torch(tensor)
        if hasattr(result, 'cpu'):
            result = result.cpu()
        if hasattr(result, 'numpy'):
            result = result.numpy()
        if hasattr(result, 'detach'):
            result = result.detach().numpy()
        if isinstance(result, np.ndarray):
            return result
        return np.array(result, dtype=np.float32)
    except Exception as e:
        print(f"       Warning: Could not convert to numpy: {e}")
        return None


# --- Mock Classes ---
class MockConfig:
    """Mock configuration for Segformer Image Classification"""
    def __init__(self, num_labels=1000):
        # Encoder configuration (matching mit-b0)
        self.num_encoder_blocks = 4
        self.hidden_sizes = [32, 64, 160, 256]
        self.depths = [2, 2, 2, 2]
        self.num_attention_heads = [1, 2, 5, 8]
        self.sr_ratios = [8, 4, 2, 1]
        self.mlp_ratios = [4, 4, 4, 4]
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.0
        self.attention_probs_dropout_prob = 0.0
        self.decoder_hidden_size = 256
        
        # Classification configuration
        self.num_labels = num_labels
        self.use_return_dict = True
        
        # Image configuration
        self.image_size = 224
        self.num_channels = 3
        self.patch_sizes = [7, 3, 3, 3]
        self.strides = [4, 2, 2, 2]


class MockLinear:
    """Simulates torch.nn.Linear"""
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
        self.bias = np.random.randn(out_features).astype(np.float32) * 0.01
        self.in_features = in_features
        self.out_features = out_features


class MockSegformerForImageClassification:
    """Mock Segformer for Image Classification model for parameter extraction"""
    def __init__(self, config):
        self.config = config
        
        # Segformer encoder backbone
        self.segformer = MockSegformerModel(config)
        
        # Classification head
        # Input: hidden_sizes[-1], Output: num_labels
        self.classifier = MockLinear(config.hidden_sizes[-1], config.num_labels)


def create_custom_mesh_preprocessor(device):
    """
    Preprocessor to extract parameters from mock image classification model.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        if isinstance(model, MockSegformerForImageClassification):
            # Process Segformer encoder parameters
            parameters["segformer"] = {}
            main_model_preprocessor = custom_preprocessor_main_model(device)
            parameters["segformer"] = main_model_preprocessor(
                model.segformer, name=None, ttnn_module_args=None, convert_to_ttnn=None
            )
            
            # Process classifier parameters
            parameters["classifier"] = {}
            
            # Weight: transpose for linear operation
            # Original: [out_features, in_features]
            # Transposed: [in_features, out_features] for matmul
            parameters["classifier"]["weight"] = ttnn.to_device(
                ttnn.as_tensor(model.classifier.weight.T.copy()),
                device
            )
            
            # Bias: reshape to [1, 1, 1, out_features] for broadcasting
            parameters["classifier"]["bias"] = ttnn.to_device(
                ttnn.as_tensor(
                    model.classifier.bias.reshape(1, 1, 1, -1)
                ),
                device
            )
        
        return parameters
    
    return preprocessor


def pad_input_tensor(input_tensor, device):
    """
    Pad input tensor to meet conv2d requirements.
    
    Args:
        input_tensor: Input tensor [batch, channels, height, width]
        device: Device handle
    
    Returns:
        Padded tensor
    """
    CONV2D_MIN_CHANNEL_SIZE = 8
    
    shape = input_tensor.shape
    channels = shape[1] if len(shape) == 4 else shape[3]
    
    # Check if padding is needed
    if channels < CONV2D_MIN_CHANNEL_SIZE:
        # Pad to minimum channel size
        if len(shape) == 4:
            # NCHW format
            pad_size = CONV2D_MIN_CHANNEL_SIZE - channels
            # Create padded tensor
            padded_shape = (shape[0], CONV2D_MIN_CHANNEL_SIZE, shape[2], shape[3])
        else:
            # NHWC format
            pad_size = CONV2D_MIN_CHANNEL_SIZE - channels
            padded_shape = (shape[0], shape[1], shape[2], CONV2D_MIN_CHANNEL_SIZE)
        
        input_tensor = ttnn.pad(input_tensor, padded_shape, (0, 0, 0, 0), 0)
        
    elif channels > CONV2D_MIN_CHANNEL_SIZE and channels % 32 != 0:
        # Pad to multiple of 32
        new_channels = (channels + 31) // 32 * 32
        
        if len(shape) == 4:
            padded_shape = (shape[0], new_channels, shape[2], shape[3])
        else:
            padded_shape = (shape[0], shape[1], shape[2], new_channels)
        
        input_tensor = ttnn.pad(input_tensor, padded_shape, (0, 0, 0, 0), 0)
    
    return input_tensor


def test_segformer_image_classification(device):
    """Test Segformer Image Classification - full model test"""
    
    test_name = "ImageClassification | Full Model | 1000 classes"
    print(f"\n[TEST] {test_name}")
    
    try:
        np.random.seed(42)
        batch_size = 1
        num_labels = 1000
        image_size = 224
        num_channels = 3
        
        # Create configuration
        config = MockConfig(num_labels=num_labels)
        print(f"       Image size: {image_size}x{image_size}")
        print(f"       Num channels: {num_channels}")
        print(f"       Num labels: {num_labels}")
        print(f"       Hidden sizes: {config.hidden_sizes}")
        
        # Create mock input image tensor [batch, channels, height, width] - NCHW format
        input_np = np.random.randn(batch_size, num_channels, image_size, image_size).astype(np.float32)
        print(f"\n       Input shape (NCHW): {input_np.shape}")
        
        # Convert to ttnn tensor
        ttnn_input = ttnn.as_tensor(input_np)
        ttnn_input = ttnn.to_device(ttnn_input, device)
        
        # Pad if necessary (matching original TT-Metal logic)
        ttnn_input = pad_input_tensor(ttnn_input, device)
        
        if hasattr(ttnn_input, 'shape'):
            padded_shape = tuple(ttnn_input.shape)
        else:
            padded_shape = tuple(ttnn_input.data.shape)
        print(f"       Padded input shape: {padded_shape}")
        
        # Create mock model for parameter extraction
        mock_model = MockSegformerForImageClassification(config)
        
        # Extract parameters
        preprocessor = create_custom_mesh_preprocessor(device)
        parameters = preprocessor(mock_model, None, None, None)
        
        # Create Polaris model
        print(f"\n       Creating TtSegformerForImageClassification...")
        ttnn_model = TtSegformerForImageClassification(config, parameters)
        
        # Run forward pass
        print(f"       Running forward pass...")
        ttnn_output = ttnn_model(
            device,
            ttnn_input,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        
        # Get logits shape
        logits = ttnn_output.logits
        if hasattr(logits, 'shape'):
            actual_shape = tuple(logits.shape)
        else:
            actual_shape = tuple(logits.data.shape)
        
        print(f"\n       Output logits shape: {actual_shape}")
        
        # Expected shape: [batch, num_labels]
        expected_shape = (batch_size, num_labels)
        print(f"       Expected shape: {expected_shape}")
        
        # Validate output
        output_np = tensor_to_numpy(logits)
        if output_np is not None:
            if np.any(np.isnan(output_np)) or np.any(np.isinf(output_np)):
                print(f"       WARNING: Output contains NaN or Inf values")
            else:
                mean_val = np.mean(np.abs(output_np))
                print(f"       Output mean absolute value: {mean_val:.6f}")
                
                # Get predicted class
                predicted_class = np.argmax(output_np.flatten())
                print(f"       Predicted class: {predicted_class}")
        
        # Check shape (may be [batch, 1, 1, num_labels] or [batch, num_labels])
        shape_match = (
            actual_shape == expected_shape or 
            actual_shape == (batch_size, 1, 1, num_labels) or
            actual_shape == (batch_size, 1, num_labels)
        )
        
        if shape_match:
            print(f"\n[PASS] {test_name}")
            return True
        else:
            print(f"\n[FAIL] {test_name}")
            print(f"       Expected shape: {expected_shape}")
            print(f"       Got shape: {actual_shape}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_classifier_head_only(device):
    """Test just the classifier head operations"""
    
    test_name = "ImageClassification | Classifier Head Only"
    print(f"\n[TEST] {test_name}")
    
    try:
        np.random.seed(42)
        batch_size = 1
        num_labels = 1000
        hidden_size = 256  # Final encoder hidden size
        seq_len = 49  # 7x7 spatial after all downsampling
        
        config = MockConfig(num_labels=num_labels)
        
        print(f"       Batch size: {batch_size}")
        print(f"       Sequence length: {seq_len}")
        print(f"       Hidden size: {hidden_size}")
        print(f"       Num labels: {num_labels}")
        
        # Create mock encoder output [batch, seq_len, hidden_size]
        encoder_output_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        encoder_output = create_polaris_tensor(encoder_output_np, device)
        
        print(f"\n       Encoder output shape: [{batch_size}, {seq_len}, {hidden_size}]")
        
        # Create classifier weights
        classifier_weight = np.random.randn(hidden_size, num_labels).astype(np.float32) * 0.02
        classifier_bias = np.random.randn(1, 1, 1, num_labels).astype(np.float32) * 0.01
        
        classifier_weight_tensor = create_polaris_tensor(classifier_weight, device)
        classifier_bias_tensor = create_polaris_tensor(classifier_bias, device)
        
        # Step 1: Reshape (already in correct shape)
        sequence_output = encoder_output
        
        # Step 2: Global average pooling
        pooled = ttnn.mean(sequence_output, dim=1, keepdim=True)
        print(f"       After mean pooling: {tuple(pooled.shape)}")
        
        # Step 3: Squeeze
        squeezed = ttnn.squeeze(pooled, dim=1)
        print(f"       After squeeze: {tuple(squeezed.shape)}")
        
        # Step 4: Linear classification
        logits = ttnn.linear(
            squeezed,
            classifier_weight_tensor,
            bias=classifier_bias_tensor,
        )
        
        if hasattr(logits, 'shape'):
            actual_shape = tuple(logits.shape)
        else:
            actual_shape = tuple(logits.data.shape)
        
        print(f"       Logits shape: {actual_shape}")
        
        # Expected: [batch, num_labels] or similar
        expected_shapes = [
            (batch_size, num_labels),
            (batch_size, 1, num_labels),
            (batch_size, 1, 1, num_labels),
        ]
        
        shape_match = actual_shape in expected_shapes
        
        if shape_match:
            print(f"\n[PASS] {test_name}")
            return True
        else:
            print(f"\n[FAIL] {test_name}")
            print(f"       Expected one of: {expected_shapes}")
            print(f"       Got: {actual_shape}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_output_dataclass(device):
    """Test that the output dataclass works correctly"""
    
    test_name = "ImageClassification | Output Dataclass"
    print(f"\n[TEST] {test_name}")
    
    try:
        # Create mock logits tensor
        mock_logits = create_polaris_tensor(
            np.random.randn(1, 1000).astype(np.float32), device
        )
        
        # Test dataclass creation
        output = TtSegFormerImageClassifierOutput(
            loss=None,
            logits=mock_logits,
            hidden_states=None,
            attentions=None,
        )
        
        # Verify fields
        assert output.loss is None, "Loss should be None"
        assert output.logits is not None, "Logits should not be None"
        assert output.hidden_states is None, "Hidden states should be None"
        assert output.attentions is None, "Attentions should be None"
        
        print(f"       Output dataclass created successfully")
        print(f"       - loss: {output.loss}")
        print(f"       - logits shape: {tuple(output.logits.shape)}")
        print(f"       - hidden_states: {output.hidden_states}")
        print(f"       - attentions: {output.attentions}")
        
        print(f"\n[PASS] {test_name}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_input_padding(device):
    """Test input padding logic"""
    
    test_name = "ImageClassification | Input Padding"
    print(f"\n[TEST] {test_name}")
    
    try:
        batch_size = 1
        height = 224
        width = 224
        
        test_cases = [
            (3, 8, "3 channels -> 8 (min size)"),
            (8, 8, "8 channels -> 8 (no change)"),
            (16, 32, "16 channels -> 32 (round to 32)"),
            (32, 32, "32 channels -> 32 (no change)"),
            (48, 64, "48 channels -> 64 (round to 32)"),
        ]
        
        all_passed = True
        
        for channels, expected_channels, description in test_cases:
            input_np = np.random.randn(batch_size, channels, height, width).astype(np.float32)
            input_tensor = create_polaris_tensor(input_np, device)
            
            padded = pad_input_tensor(input_tensor, device)
            
            if hasattr(padded, 'shape'):
                padded_shape = tuple(padded.shape)
            else:
                padded_shape = tuple(padded.data.shape)
            
            actual_channels = padded_shape[1]
            
            if actual_channels == expected_channels:
                print(f"       [OK] {description}: {channels} -> {actual_channels}")
            else:
                print(f"       [FAIL] {description}: expected {expected_channels}, got {actual_channels}")
                all_passed = False
        
        if all_passed:
            print(f"\n[PASS] {test_name}")
            return True
        else:
            print(f"\n[FAIL] {test_name}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests(name, device, cfg=None, **kwargs):
    """
    Run all Segformer Image Classification tests.
    
    Args:
        device: Device handle from Polaris
        cfg: Configuration dict with bs, image_height, image_width, num_labels
        **kwargs: Additional arguments passed by Polaris
    """
    print("\n" + "=" * 80)
    print("=== Polaris Segformer Image Classification Tests ===")
    print("=" * 80)
    
    # Extract config parameters if provided
    if cfg:
        bs = cfg.get('bs', 1)
        image_height = cfg.get('image_height', 224)
        image_width = cfg.get('image_width', 224)
        num_labels = cfg.get('num_labels', 1000)
        print(f"Config: bs={bs}, image={image_height}x{image_width}, labels={num_labels}")
    
    results = []
    
    results.append(test_output_dataclass(device))
    results.append(test_classifier_head_only(device))
    # results.append(test_input_padding(device))
    # results.append(test_segformer_image_classification(device))
    
    passed_count = sum(results)
    total_tests = len(results)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{total_tests} passed")
    print("=" * 80)
    
    return passed_count == total_tests

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    print(f"Device opened: {device}")
    
    try:
        all_passed = run_all_tests("local_test", device, cfg=None)
        sys.exit(0 if all_passed else 1)
    finally:
        ttnn.close_device(device)
        print("Device closed.")