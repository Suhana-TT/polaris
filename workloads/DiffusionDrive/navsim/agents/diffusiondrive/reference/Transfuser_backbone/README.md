# TransFuser Backbone Validation Tests

This directory contains comprehensive validation scripts for all modules in the TransFuser backbone conversion from PyTorch to TTSIM.

## Test Suite Overview

All tests follow the same workflow:
1. **Create test data** - Random numpy arrays with appropriate shapes
2. **PyTorch forward pass** - Reference implementation for ground truth
3. **TTSIM forward pass** - Converted implementation
4. **Weight injection** - Copy PyTorch weights to TTSIM modules
5. **Shape validation** - Verify output shapes match
6. **Data validation** - Verify numerical equivalence
7. **Report results** - Pass/warn/fail with difference metrics

## Test Files

### test_1_selfattention.py
**Module**: `SelfAttention`
**Purpose**: Multi-head self-attention layer
**Key operations**: Q/K/V projections, scaled dot-product attention, output projection
**Expected max diff**: < 1e-4

**Run**:
```bash
python test_1_selfattention.py
```

### test_2_block.py
**Module**: `Block`
**Purpose**: Transformer block (attention + MLP with residual connections)
**Key operations**: LayerNorm, SelfAttention, feedforward MLP, residual adds
**Expected max diff**: < 1e-4

**Run**:
```bash
python test_2_block.py
```

### test_3_gpt.py
**Module**: `GPT`
**Purpose**: Full GPT transformer for multi-modal fusion
**Key operations**: Positional embeddings, multi-layer transformer, modality splitting
**Expected max diff**: < 1e-4

**Run**:
```bash
python test_3_gpt.py
```

### test_4_multihead_attention.py
**Module**: `MultiheadAttentionWithAttention`
**Purpose**: Multi-head attention that returns attention weights
**Key operations**: Cross-attention with separate Q/K/V inputs, attention weight averaging
**Expected max diff**: < 1e-4

**Run**:
```bash
python test_4_multihead_attention.py
```

### test_5_decoder_layer.py
**Module**: `TransformerDecoderLayerWithAttention`
**Purpose**: Single transformer decoder layer with self and cross attention
**Key operations**: Self-attention, cross-attention, feedforward, layer norms
**Expected max diff**: < 1e-4

**Run**:
```bash
python test_5_decoder_layer.py
```

### test_6_decoder.py
**Module**: `TransformerDecoderWithAttention`
**Purpose**: Full transformer decoder stack
**Key operations**: Multiple decoder layers, attention averaging across layers
**Expected max diff**: < 1e-4

**Run**:
```bash
python test_6_decoder.py
```

### test_7_transfuser_backbone.py
**Module**: `TransfuserBackbone`
**Purpose**: Complete multi-scale fusion model (image + LiDAR)
**Key operations**: timm encoders, GPT fusion, FPN top-down pathway
**Expected result**: Shape consistency, successful forward pass

**Note**: This test validates shape and execution, not full numerical equivalence, because:
- timm encoders remain as PyTorch (pretrained models)
- Full end-to-end weight injection is complex
- Primary goal is to verify TTSIM transformer components work correctly

**Run**:
```bash
python test_7_transfuser_backbone.py
```

## Running All Tests

### Run individually:
```bash
python test_1_selfattention.py
python test_2_block.py
python test_3_gpt.py
python test_4_multihead_attention.py
python test_5_decoder_layer.py
python test_6_decoder.py
python test_7_transfuser_backbone.py
```

### Run all at once (PowerShell):
```powershell
Get-ChildItem test_*.py | ForEach-Object { python $_.Name }
```

### Run all at once (Bash):
```bash
for test in test_*.py; do python "$test"; done
```

## Validation Criteria

### Thresholds
- **✓ PASS**: max_diff < 1e-4 (excellent agreement)
- **⚠ WARN**: max_diff < 1e-3 (acceptable agreement)
- **✗ FAIL**: max_diff >= 1e-3 (needs investigation)

### Common Issues
1. **Missing bias**: Linear layers require separate F.Bias operations
2. **Dropout eval mode**: Must use positional args: `F.Dropout(name, prob, False, module=...)`
3. **Data computation**: Ensure `try_compute_data` is called in shape inference functions
4. **Weight transpose**: TTSIM Linear expects transposed weights (F vs C layout)
5. **Shape mismatches**: Check tensor reshaping and permutation operations

## Conversion Notes

### Key TTSIM Patterns
- `nn.Linear` → `F.Linear` + `F.Bias`
- `nn.Parameter` → `F._from_shape(..., is_param=True)`
- `torch.cat` → `F.ConcatX`
- `F.interpolate(mode='bilinear')` → `F.Resize(mode='linear')`
- `nn.Dropout(p, train=False)` → `F.Dropout(name, p, False, module=...)`
- Arithmetic ops → `F.Add`, `F.Multiply`, etc.

### Weight Injection Pattern
```python
# Linear + Bias
ttsim_linear.params[0][1].data = pytorch_linear.weight.data.T.numpy()
ttsim_bias.params[0][1].data = pytorch_linear.bias.data.numpy()

# LayerNorm
ttsim_ln.params[0][1].data = pytorch_ln.weight.data.numpy()
ttsim_ln.params[1][1].data = pytorch_ln.bias.data.numpy()

# Parameter
ttsim_param.data = pytorch_param.data.numpy()
```

## Test Results Summary

| Module | Test File | Status | Max Diff |
|--------|-----------|--------|----------|
| SelfAttention | test_1 | ✓ PASS | < 1e-7 |
| Block | test_2 | ✓ PASS | < 1e-6 |
| GPT | test_3 | ✓ PASS | < 1e-6 |
| MultiheadAttentionWithAttention | test_4 | Pending | - |
| TransformerDecoderLayerWithAttention | test_5 | Pending | - |
| TransformerDecoderWithAttention | test_6 | Pending | - |
| TransfuserBackbone | test_7 | Pending | Shape only |

## Troubleshooting

### Import errors
- Ensure `PYTHONPATH` includes workspace root
- Check that `navsim` and `ttsim` are accessible

### Missing nuplan
- Install: `pip install nuplan-devkit`
- Or mock TransfuserConfig if only testing components

### timm download issues
- Ensure internet connection for pretrained model downloads
- Or use `pretrained=False` for architecture testing only

### Data is None
- Check `try_compute_data` calls in ttsim operations
- Verify shape inference functions call data computation

## Contact

For issues or questions about these tests, refer to the main project documentation or file an issue.
