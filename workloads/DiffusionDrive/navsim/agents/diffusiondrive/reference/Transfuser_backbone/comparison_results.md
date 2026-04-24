# Comparison Results (Transfuser_backbone) -- 2026-04-20 19:51:11

## test_1_selfattention.py  --  PASS

### stdout

```
======================================================================
TEST: SelfAttention - Multi-head Self-Attention
======================================================================

--- PyTorch SelfAttention ---
Input shape: torch.Size([2, 64, 128])
Output shape: torch.Size([2, 64, 128])
Output stats: min=-0.206105, max=0.221421, mean=0.003948

--- TTSIM SelfAttention ---
Injecting weights...
Input shape: Shape([2, 64, 128])
Output shape: Shape([2, 64, 128])
Output stats: min=-0.206105, max=0.221421, mean=0.003948

--- Numerical Comparison ---
Max difference: 0.0000001490
Mean difference: 0.0000000181
Std difference: 0.0000000161
PASS: PASS: SelfAttention matches PyTorch (max diff < 1e-4)
```

---

## test_2_block.py  --  PASS

### stdout

```
======================================================================
TEST: Block - Transformer Block (Attention + MLP)
======================================================================

--- PyTorch Block ---
Input shape: torch.Size([2, 64, 128])
Output shape: torch.Size([2, 64, 128])
Output stats: min=-3.558093, max=4.363064, mean=0.004409

--- TTSIM Block ---
Injecting weights...
Input shape: Shape([2, 64, 128])
Output shape: Shape([2, 64, 128])
Output stats: min=-3.558093, max=4.363064, mean=0.004409

--- Numerical Comparison ---
Max difference: 0.0000007749
Mean difference: 0.0000000694
Std difference: 0.0000000703
PASS: PASS: Block matches PyTorch (max diff < 1e-4)
```

---

## test_3_gpt.py  --  PASS

### stdout

```
======================================================================
TEST: GPT - Full Transformer (Multi-modal Fusion)
======================================================================

--- PyTorch GPT ---
Image input: torch.Size([2, 128, 8, 32]) -> output: torch.Size([2, 128, 8, 32])
LiDAR input: torch.Size([2, 128, 8, 8]) -> output: torch.Size([2, 128, 8, 8])
Image output stats: min=-4.244159, max=3.795542
LiDAR output stats: min=-4.366917, max=3.855736

--- TTSIM GPT ---
Injecting weights...
Image input: Shape([2, 128, 8, 32]) -> output: Shape([2, 128, 8, 32])
LiDAR input: Shape([2, 128, 8, 8]) -> output: Shape([2, 128, 8, 8])
Image output stats: min=-4.244158, max=3.795542
LiDAR output stats: min=-4.366917, max=3.855736

--- Numerical Comparison ---
Image - Max diff: 0.0000009537, Mean diff: 0.0000001138
LiDAR - Max diff: 0.0000009537, Mean diff: 0.0000001133
PASS: PASS: GPT matches PyTorch (max diff < 1e-4)
```

---

## test_4_multihead_attention.py  --  PASS

### stdout

```
======================================================================
TEST: MultiheadAttentionWithAttention
======================================================================

--- PyTorch MultiheadAttentionWithAttention ---
Query shape: torch.Size([2, 32, 256])
Key shape: torch.Size([2, 64, 256])
Value shape: torch.Size([2, 64, 256])
Output shape: torch.Size([2, 32, 256])
Attention shape: torch.Size([2, 32, 64])
Output stats: min=-0.223831, max=0.236379

--- TTSIM MultiheadAttentionWithAttention ---
Injecting weights...
Query shape: Shape([2, 32, 256])
Key shape: Shape([2, 64, 256])
Value shape: Shape([2, 64, 256])
Output shape: Shape([2, 32, 256])
Attention shape: Shape([2, 32, 64])
Output stats: min=-0.223831, max=0.236379

--- Numerical Comparison ---
Output - Max diff: 0.0000001788, Mean diff: 0.0000000197
Attention - Max diff: 0.0000000056, Mean diff: 0.0000000008

(atol=0.0001, rtol=0.0001):
  Output:    PASS
  Attention: PASS

======================================================================
OVERALL: PASS: PASS - (atol=0.0001, rtol=0.0001)
======================================================================
```

---

## test_5_decoder_layer.py  --  PASS

### stdout

```
======================================================================
TEST: TransformerDecoderLayerWithAttention
======================================================================

--- PyTorch TransformerDecoderLayerWithAttention ---
Target shape: torch.Size([2, 32, 256])
Memory shape: torch.Size([2, 64, 256])
Output shape: torch.Size([2, 32, 256])
Attention shape: torch.Size([2, 32, 64])
Output stats: min=-4.038055, max=3.674921

--- TTSIM TransformerDecoderLayerWithAttention ---
Injecting weights...
Target shape: Shape([2, 32, 256])
Memory shape: Shape([2, 64, 256])
Output shape: Shape([2, 32, 256])
Attention shape: Shape([2, 32, 64])
Output stats: min=-4.038055, max=3.674921

--- Numerical Comparison ---
Output - Max diff: 0.0000009537, Mean diff: 0.0000001159
Attention - Max diff: 0.0000000056, Mean diff: 0.0000000009
PASS: PASS: TransformerDecoderLayerWithAttention matches PyTorch (max diff < 1e-4)
```

---

## test_6_decoder.py  --  PASS

### stdout

```
======================================================================
TEST: TransformerDecoderWithAttention
======================================================================

--- PyTorch TransformerDecoderWithAttention ---
Queries shape: torch.Size([2, 32, 256])
Memory shape: torch.Size([2, 64, 256])
Output shape: torch.Size([2, 32, 256])
Attention shape: torch.Size([2, 32, 64])
Output stats: min=-3.843951, max=3.726617

--- TTSIM TransformerDecoderWithAttention ---
Injecting weights...
Queries shape: Shape([2, 32, 256])
Memory shape: Shape([2, 64, 256])
Output shape: Shape([2, 32, 256])
Attention shape: Shape([2, 32, 64])
Output stats: min=-3.843952, max=3.726618

--- Numerical Comparison ---
Output - Max diff: 0.0000014305, Mean diff: 0.0000002102
Attention - Max diff: 0.0000000037, Mean diff: 0.0000000007

(atol=0.0001, rtol=0.0001):
  Output:    PASS
  Attention: PASS

======================================================================
OVERALL: PASS: PASS - (atol=0.0001, rtol=0.0001)
======================================================================
```

---

## test_7_transfuser_backbone.py  --  PASS

### stdout

```
======================================================================
TEST: TransfuserBackbone
======================================================================

--- PyTorch TransfuserBackbone ---
Image input shape: [1, 3, 224, 224]
LiDAR input shape: [1, 1, 256, 256]
BEV features shape: [1, 64, 64, 64]
BEV features stats: min=0.000000, max=0.274950
Fused features shape: [1, 512]

--- TTSIM TransfuserBackbone ---
Injecting weights...
Image input shape: [1, 3, 224, 224]
LiDAR input shape: [1, 1, 256, 256]
Transformer output shape (image): [1, 512, 8, 32]
Transformer output shape (lidar): [1, 512, 8, 8]
BEV features shape: [1, 64, 64, 64]
BEV features stats: min=0.000000, max=0.274950
Fused features shape: [1, 512]

--- Numerical Comparison ---
Transformer Layer 0 - Image: Max diff: 0.0000009537, Mean diff: 0.0000000693
Transformer Layer 0 - LiDAR: Max diff: 0.0000007153, Mean diff: 0.0000000895
Transformer Layer 1 - Image: Max diff: 0.0000009537, Mean diff: 0.0000000615
Transformer Layer 1 - LiDAR: Max diff: 0.0000009537, Mean diff: 0.0000002006
Transformer Layer 2 - Image: Max diff: 0.0000014305, Mean diff: 0.0000000940
Transformer Layer 2 - LiDAR: Max diff: 0.0000023842, Mean diff: 0.0000003839
Transformer Layer 3 - Image: Max diff: 0.0000052452, Mean diff: 0.0000005301
Transformer Layer 3 - LiDAR: Max diff: 0.0000050068, Mean diff: 0.0000006594
BEV features - Max diff: 0.0000003725, Mean diff: 0.0000000231
Fused features - Max diff: 0.0000009537, Mean diff: 0.0000000927

(atol=0.0001, rtol=0.0001):
  Transformer Layer 0: PASS
  Transformer Layer 1: PASS
  Transformer Layer 2: PASS
  Transformer Layer 3: PASS
  BEV features:          PASS
  Fused features:        PASS

======================================================================
OVERALL: PASS: PASS - (atol=0.0001, rtol=0.0001)
======================================================================
```

---


