# Comparison Results (Conditional_unet1d) -- 2026-04-10 12:47:51

## test_0_sinusoidal_pos_emb.py  --  PASS

### stdout

```
======================================================================
SinusoidalPosEmb Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch SinusoidalPosEmb ---
Input (timesteps) shape: torch.Size([4])
Timesteps: [102 435 860 270]
Output shape: (4, 256)
Output stats: min=-0.999997, max=0.999994, mean=0.247922

--- TTSIM SinusoidalPosEmb ---
Input (timesteps) shape: Shape([4])
Output shape: Shape([4, 256])
Output stats: min=-0.999997, max=0.999994, mean=0.247922

--- Numerical Comparison ---
Tolerance: atol=0.0001, rtol=0.0001
  Max absolute difference:  0.0000292666
  Mean absolute difference: 0.0000012467

  Shape match: [PASS]  PT=(4, 256)  TTSIM=Shape([4, 256])

======================================================================
OVERALL: [PASS] PASS - TTSIM matches PyTorch
======================================================================

```

---

## test_1_conv1d_block.py  --  PASS

### stdout

```
======================================================================
Conv1dBlock Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch Conv1dBlock ---
Input shape:  (2, 64, 16)
Output shape: (2, 128, 16)
Output stats: min=-0.308843, max=3.934181, mean=0.239659

--- TTSIM Conv1dBlock ---
Injecting weights...
Input shape:  Shape([2, 64, 16])
Output shape: Shape([2, 128, 16])
Output stats: min=-0.308843, max=3.934180, mean=0.239659

--- Numerical Comparison ---
Tolerance: atol=0.0001, rtol=0.0001
  Max absolute difference:  0.0000021458
  Mean absolute difference: 0.0000001199

  Shape match: [PASS]  PT=(2, 128, 16)  TTSIM=Shape([2, 128, 16])

======================================================================
OVERALL: [PASS] PASS - TTSIM matches PyTorch
======================================================================

```

---

## test_2_downsample1d.py  --  PASS

### stdout

```
======================================================================
Downsample1d Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch Downsample1d ---
Input shape:  (2, 128, 16)
Output shape: (2, 128, 8)
Output stats: min=-1.981432, max=1.961904, mean=0.018651

--- TTSIM Downsample1d ---
Injecting weights...
Input shape:  Shape([2, 128, 16])
Output shape: Shape([2, 128, 8])
Output stats: min=-1.981432, max=1.961904, mean=0.018651

--- Numerical Comparison ---
Tolerance: atol=0.0001, rtol=0.0001
  Max absolute difference:  0.0000011921
  Mean absolute difference: 0.0000001463

  Shape match: [PASS]  PT=(2, 128, 8)  TTSIM=Shape([2, 128, 8])

======================================================================
OVERALL: [PASS] PASS - TTSIM matches PyTorch
======================================================================

```

---

## test_3_upsample1d.py  --  PASS

### stdout

```
======================================================================
Upsample1d Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch Upsample1d ---
Input shape:  (2, 128, 8)
Output shape: (2, 128, 16)
Output stats: min=-1.409278, max=1.866521, mean=0.003510

--- TTSIM Upsample1d ---
Injecting weights...
Input shape:  Shape([2, 128, 8])
Output shape: Shape([2, 128, 16])
Output stats: min=-1.409279, max=1.866520, mean=0.003510

--- Numerical Comparison ---
Tolerance: atol=0.0001, rtol=0.0001
  Max absolute difference:  0.0000004470
  Mean absolute difference: 0.0000000603

  Shape match: [PASS]  PT=(2, 128, 16)  TTSIM=Shape([2, 128, 16])

======================================================================
OVERALL: [PASS] PASS - TTSIM matches PyTorch
======================================================================

```

---

## test_4_cond_residual_block.py  --  PASS

### stdout

```
======================================================================
ConditionalResidualBlock1D Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch ConditionalResidualBlock1D ---
Input x shape:    (2, 64, 16)
Input cond shape: (2, 256)
Output shape:     (2, 128, 16)
Output stats: min=-2.184460, max=3.862171, mean=0.237125

--- TTSIM ConditionalResidualBlock1D ---
Injecting weights...
Output shape: Shape([2, 128, 16])
Output stats: min=-2.184460, max=3.862172, mean=0.237125

--- Numerical Comparison ---
Tolerance: atol=0.0001, rtol=0.0001
  Max absolute difference:  0.0000026226
  Mean absolute difference: 0.0000002334

  Shape match: [PASS]  PT=(2, 128, 16)  TTSIM=Shape([2, 128, 16])

======================================================================
OVERALL: [PASS] PASS - TTSIM matches PyTorch
======================================================================


======================================================================
Test 2: in_channels == out_channels (identity residual)
======================================================================
  Shape: PT=(2, 128, 16)  TTSIM=Shape([2, 128, 16])  match=[PASS]
  Max diff: 0.0000028610

======================================================================
TEST 2: [PASS] PASS
======================================================================

```

---

## test_5_conditional_unet1d.py  --  PASS

### stdout

```
======================================================================
ConditionalUnet1D Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch ConditionalUnet1D ---
Input sample shape:  (2, 8, 4)
Timestep:            [ 5 10]
Global cond shape:   (2, 16)
Output shape:        (2, 8, 4)
Output stats: min=-0.855385, max=0.807097, mean=0.057971

--- TTSIM ConditionalUnet1D ---
Injecting weights...
[PASS] Weight injection complete
Output shape: Shape([2, 8, 4])
Output stats: min=-0.855385, max=0.807097, mean=0.057971

--- Numerical Comparison ---
Tolerance: atol=0.001, rtol=0.001
  Max absolute difference:  0.0000005629
  Mean absolute difference: 0.0000001898

  Shape match: [PASS]  PT=(2, 8, 4)  TTSIM=Shape([2, 8, 4])

======================================================================
OVERALL: [PASS] PASS - TTSIM matches PyTorch
======================================================================

```

---
