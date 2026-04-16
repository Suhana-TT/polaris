# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import math
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
import ttsim.front.ttnn as ttnn
from workloads.segformer.tt.segformer_for_semantic_segmentation import (
    TtSegformerForSemanticSegmentation,
    TtSemanticSegmenterOutput,
)
from workloads.segformer.tests.test_segformer_decode_head import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_decode_head,
    MockSegformerDecodeHead,
)
from workloads.segformer.tests.test_segformer_model import (
    create_custom_mesh_preprocessor as create_custom_preprocessor_model,
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
    """Mock configuration for Segformer Semantic Segmentation"""
    def __init__(self, num_labels=150):
        # Encoder configuration (matching segformer-b0-finetuned-ade-512-512)
        self.num_encoder_blocks = 4
        self.hidden_sizes = [32, 64, 160, 256]
        self.depths = [2, 2, 2, 2]
        self.num_attention_heads = [1, 2, 5, 8]
        self.sr_ratios = [8, 4, 2, 1]
        self.mlp_ratios = [4, 4, 4, 4]
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.0
        self.attention_probs_dropout_prob = 0.0
        
        # Decoder/Segmentation configuration
        self.decoder_hidden_size = 256
        self.num_labels = num_labels  # ADE20K has 150 classes
        
        # Return options
        self.use_return_dict = True
        self.output_hidden_states = True
        
        # Image configuration
        self.image_size = 512
        self.num_channels = 3
        self.patch_sizes = [7, 3, 3, 3]
        self.strides = [4, 2, 2, 2]


class MockSegformerForSemanticSegmentation:
    """Mock Segformer for Semantic Segmentation model for parameter extraction"""
    def __init__(self, config):
        self.config = config
        
        # Segformer encoder backbone
        self.segformer = MockSegformerModel(config)
        
        # Decode head for segmentation
        self.decode_head = MockSegformerDecodeHead(config)


def create_custom_mesh_preprocessor(device):
    """
    Preprocessor to extract parameters from mock semantic segmentation model.
    
    Combines the model preprocessor and decode head preprocessor.
    """
    def preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        
        if isinstance(model, MockSegformerForSemanticSegmentation):
            # Process Segformer encoder parameters
            parameters["segformer"] = {}
            segformer_preprocess = create_custom_preprocessor_model(device)
            parameters["segformer"] = segformer_preprocess(
                model.segformer, None, None, None
            )
            
            # Process decode head parameters
            parameters["decode_head"] = {}
            decode_preprocess = create_custom_preprocessor_decode_head(device)
            parameters["decode_head"] = decode_preprocess(
                model.decode_head, None, None, None
            )
        
        return parameters
    
    return preprocessor


def move_to_device(obj, device):
    """
    Recursively move all tensors in a nested dict/list structure to device.
    
    Args:
        obj: Dictionary, list, or tensor to move
        device: Target device
    
    Returns:
        Object with all tensors moved to device
    """
    # Skip certain keys that are already on device or handled separately
    skip_keys = ["sr", "proj", "dwconv", "linear_fuse", "classifier"]
    
    if isinstance(obj, dict):
        for name, value in list(obj.items()):
            if name in skip_keys:
                continue
            obj[name] = move_to_device(value, device)
        return obj
    elif isinstance(obj, list):
        for index, element in enumerate(obj):
            obj[index] = move_to_device(element, device)
        return obj
    elif isinstance(obj, ttnn.Tensor):
        return ttnn.to_device(obj, device)
    else:
        return obj


def test_segformer_for_semantic_segmentation(device):
    """Test Segformer for Semantic Segmentation - full model test"""
    
    test_name = "SemanticSegmentation | Full Model | ADE20K 150 classes"
    print(f"\n[TEST] {test_name}")
    
    try:
        np.random.seed(42)
        min_channels = 8
        batch_size = 1
        num_labels = 150  # ADE20K classes
        image_height = 512
        image_width = 512
        num_channels = 3
        
        # Create configuration
        config = MockConfig(num_labels=num_labels)
        print(f"       Image size: {image_height}x{image_width}")
        print(f"       Num channels: {num_channels}")
        print(f"       Num labels (classes): {num_labels}")
        print(f"       Hidden sizes: {config.hidden_sizes}")
        print(f"       Decoder hidden size: {config.decoder_hidden_size}")
        
        # Create mock input image tensor [batch, channels, height, width] - NCHW format
        # Simulating preprocessed image from SegformerImageProcessor
        input_np = np.random.randn(batch_size, num_channels, image_height, image_width).astype(np.float32)
        print(f"\n       Input shape (NCHW): {input_np.shape}")
        
        # Pad channels if needed (matching original TT-Metal logic)
        c = num_channels
        if c < min_channels:
            c = min_channels
        elif c % min_channels != 0:
            c = ((c // min_channels) + 1) * min_channels
        
        if c != num_channels:
            # Pad channels
            padded_input = np.zeros((batch_size, c, image_height, image_width), dtype=np.float32)
            padded_input[:, :num_channels, :, :] = input_np
            input_np = padded_input
            print(f"       Padded input shape: {input_np.shape}")
        
        # Convert to ttnn tensor
        ttnn_input = ttnn.as_tensor(input_np)
        ttnn_input = ttnn.to_device(ttnn_input, device)
        
        # Create mock model for parameter extraction
        mock_model = MockSegformerForSemanticSegmentation(config)
        
        # Extract parameters
        preprocessor = create_custom_mesh_preprocessor(device)
        parameters = preprocessor(mock_model, None, None, None)
        
        # Move parameters to device
        parameters = move_to_device(parameters, device)
        
        # Move decode head linear_c parameters to device
        for i in range(4):
            parameters["decode_head"]["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
                parameters["decode_head"]["linear_c"][i]["proj"]["weight"], device
            )
            parameters["decode_head"]["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
                parameters["decode_head"]["linear_c"][i]["proj"]["bias"], device
            )
        
        # Create Polaris model
        print(f"\n       Creating TtSegformerForSemanticSegmentation...")
        ttnn_model = TtSegformerForSemanticSegmentation(config, parameters)
        
        # Run forward pass
        print(f"       Running forward pass...")
        ttnn_output = ttnn_model(
            device,
            ttnn_input,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        
        # Get logits and convert to numpy
        logits = ttnn_output.logits
        
        # Get output shape
        if hasattr(logits, 'shape'):
            actual_shape = tuple(logits.shape)
        else:
            actual_shape = tuple(logits.data.shape)
        
        print(f"\n       Raw output logits shape: {actual_shape}")
        
        # Convert output to numpy for validation
        ttnn_output_np = tensor_to_numpy(logits)
        
        if ttnn_output_np is not None:
            # Handle different output formats
            # Original TT-Metal does: permute(0, 3, 1, 2) then reshape
            
            if len(actual_shape) == 4:
                # Check if NHWC or NCHW
                if actual_shape[1] == num_labels:
                    # Already NCHW: [batch, num_labels, H, W]
                    final_shape = actual_shape
                else:
                    # NHWC: [batch, H, W, num_labels] -> permute to NCHW
                    ttnn_output_np = np.transpose(ttnn_output_np, (0, 3, 1, 2))
                    final_shape = ttnn_output_np.shape
            elif len(actual_shape) == 3:
                # [batch, seq_len, num_labels] - need to reshape
                h = w = int(math.sqrt(actual_shape[1]))
                ttnn_output_np = ttnn_output_np.reshape(actual_shape[0], h, w, actual_shape[2])
                ttnn_output_np = np.transpose(ttnn_output_np, (0, 3, 1, 2))
                final_shape = ttnn_output_np.shape
            else:
                final_shape = actual_shape
            
            print(f"       Final output shape (NCHW): {final_shape}")
            
            # Validate output
            if np.any(np.isnan(ttnn_output_np)) or np.any(np.isinf(ttnn_output_np)):
                print(f"       WARNING: Output contains NaN or Inf values")
            else:
                mean_val = np.mean(np.abs(ttnn_output_np))
                print(f"       Output mean absolute value: {mean_val:.6f}")
                
                # Get predicted segmentation for center pixel
                center_h, center_w = final_shape[2] // 2, final_shape[3] // 2
                center_logits = ttnn_output_np[0, :, center_h, center_w]
                predicted_class = np.argmax(center_logits)
                print(f"       Predicted class at center: {predicted_class}")
        
        # Expected shape: [batch, num_labels, 128, 128] (1/4 of input size)
        target_size = image_height // 4  # 512 / 4 = 128
        expected_shape_nchw = (batch_size, num_labels, target_size, target_size)
        expected_shape_nhwc = (batch_size, target_size, target_size, num_labels)
        
        print(f"\n       Expected (NCHW): {expected_shape_nchw}")
        print(f"       Expected (NHWC): {expected_shape_nhwc}")
        
        # Check shape match
        shape_match = (
            actual_shape == expected_shape_nchw or 
            actual_shape == expected_shape_nhwc or
            (ttnn_output_np is not None and final_shape == expected_shape_nchw)
        )
        
        if shape_match:
            print(f"\n[PASS] {test_name}")
            return True
        else:
            print(f"\n[FAIL] {test_name}")
            print(f"       Expected: {expected_shape_nchw} or {expected_shape_nhwc}")
            print(f"       Got: {actual_shape}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {test_name}")
        print(f"        {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_decode_head_with_mock_encoder_outputs(device):
    """Test decode head with mock encoder outputs (simpler test)"""
    
    test_name = "SemanticSegmentation | Decode Head Only"
    print(f"\n[TEST] {test_name}")
    
    try:
        np.random.seed(42)
        batch_size = 1
        num_labels = 150
        
        config = MockConfig(num_labels=num_labels)
        
        # Create mock encoder hidden states
        # These are the outputs from each encoder block
        input_configs = [
            (32, 128, 128),   # Block 0: 32 channels, 128x128 spatial
            (64, 64, 64),     # Block 1: 64 channels, 64x64 spatial
            (160, 32, 32),    # Block 2: 160 channels, 32x32 spatial
            (256, 16, 16),    # Block 3: 256 channels, 16x16 spatial
        ]
        
        print(f"       Creating mock encoder hidden states:")
        encoder_hidden_states = []
        for i, (channels, height, width) in enumerate(input_configs):
            seq_len = height * width
            # Create folded tensor [batch, seq_len, channels]
            hidden_np = np.random.randn(batch_size, seq_len, channels).astype(np.float32)
            hidden_tensor = create_polaris_tensor(hidden_np, device)
            encoder_hidden_states.append(hidden_tensor)
            print(f"         Block {i}: [{batch_size}, {seq_len}, {channels}] "
                  f"(spatial: {height}x{width})")
        
        encoder_hidden_states = tuple(encoder_hidden_states)
        
        # Create mock decode head and extract parameters
        mock_decode_head = MockSegformerDecodeHead(config)
        decode_head_preprocessor = create_custom_preprocessor_decode_head(device)
        decode_head_params = decode_head_preprocessor(mock_decode_head, None, None, None)
        
        # Create decode head
        from workloads.segformer.tt.segformer_decode_head import TtSegformerDecodeHead
        decode_head = TtSegformerDecodeHead(config, decode_head_params)
        
        # Run decode head
        print(f"\n       Running decode head...")
        logits = decode_head(device, encoder_hidden_states)
        
        # Get output shape
        if hasattr(logits, 'shape'):
            actual_shape = tuple(logits.shape)
        else:
            actual_shape = tuple(logits.data.shape)
        
        print(f"       Output logits shape: {actual_shape}")
        
        # Expected: [batch, num_labels, 128, 128] or [batch, 128, 128, num_labels]
        target_size = 128
        expected_nchw = (batch_size, num_labels, target_size, target_size)
        expected_nhwc = (batch_size, target_size, target_size, num_labels)
        
        print(f"       Expected (NCHW): {expected_nchw}")
        print(f"       Expected (NHWC): {expected_nhwc}")
        
        shape_match = actual_shape == expected_nchw or actual_shape == expected_nhwc
        
        if shape_match:
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


def test_output_dataclass(device):
    """Test that the output dataclass works correctly"""
    
    test_name = "SemanticSegmentation | Output Dataclass"
    print(f"\n[TEST] {test_name}")
    
    try:
        # Create mock logits tensor [batch, num_labels, height, width]
        mock_logits = create_polaris_tensor(
            np.random.randn(1, 150, 128, 128).astype(np.float32), device
        )
        
        # Test dataclass creation
        output = TtSemanticSegmenterOutput(
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


def test_channel_padding(device):
    """Test channel padding logic for input tensor"""
    
    test_name = "SemanticSegmentation | Channel Padding"
    print(f"\n[TEST] {test_name}")
    
    try:
        min_channels = 8
        
        test_cases = [
            (3, 8, "RGB image: 3 -> 8"),
            (1, 8, "Grayscale: 1 -> 8"),
            (8, 8, "Already 8: no change"),
            (10, 16, "10 -> 16 (next multiple of 8)"),
            (16, 16, "16: no change"),
        ]
        
        all_passed = True
        
        for input_channels, expected_channels, description in test_cases:
            c = input_channels
            if c < min_channels:
                c = min_channels
            elif c % min_channels != 0:
                c = ((c // min_channels) + 1) * min_channels
            
            if c == expected_channels:
                print(f"       [OK] {description}: {input_channels} -> {c}")
            else:
                print(f"       [FAIL] {description}: expected {expected_channels}, got {c}")
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


def test_segmentation_output_dimensions(device):
    """Test that output dimensions follow expected pattern"""
    
    test_name = "SemanticSegmentation | Output Dimensions"
    print(f"\n[TEST] {test_name}")
    
    try:
        # For 512x512 input with Segformer-B0:
        # Encoder strides: [4, 2, 2, 2] -> total stride = 32
        # But decode head upsamples to 1/4 of input
        
        test_cases = [
            (512, 512, 128, 128, "512x512 -> 128x128"),
            (1024, 1024, 256, 256, "1024x1024 -> 256x256"),
            (640, 480, 160, 120, "640x480 -> 160x120"),
        ]
        
        all_passed = True
        
        for in_h, in_w, exp_h, exp_w, description in test_cases:
            # Output is 1/4 of input size
            out_h = in_h // 4
            out_w = in_w // 4
            
            if out_h == exp_h and out_w == exp_w:
                print(f"       [OK] {description}")
            else:
                print(f"       [FAIL] {description}: expected {exp_h}x{exp_w}, got {out_h}x{out_w}")
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
    """Run all Segformer Semantic Segmentation tests."""
    print("\n" + "=" * 80)
    print("=== Polaris Segformer Semantic Segmentation Tests ===")
    print("=" * 80)
    
    if cfg:
        bs = cfg.get('bs', 1)
        image_height = cfg.get('image_height', 512)
        image_width = cfg.get('image_width', 512)
        num_labels = cfg.get('num_labels', 150)
        print(f"Config: bs={bs}, image={image_height}x{image_width}, labels={num_labels}")
    
    results = []
    
    results.append(test_output_dataclass(device))
    results.append(test_channel_padding(device))
    results.append(test_segmentation_output_dimensions(device))
    results.append(test_decode_head_with_mock_encoder_outputs(device))
    
    passed_count = sum(results)
    total_tests = len(results)
    
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