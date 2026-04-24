# Comparison Results (blocks) -- 2026-04-10 12:45:52

## test_1_grid_sample_bev_attention.py  --  PASS

### stdout

```
======================================================================
GridSampleCrossBEVAttention Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch GridSampleCrossBEVAttention ---
  queries shape:    (2, 6, 256)
  traj_points shape:(2, 6, 8, 2)
  bev_feature shape:(2, 64, 16, 16)
  Output shape:     (2, 6, 256)
  Output stats: min=-3.397088, max=4.273269, mean=0.018217

--- TTSIM GridSampleCrossBEVAttention ---
  Injecting weights...
  Output shape: Shape([2, 6, 256])
  Output stats: min=-3.397088, max=4.273269, mean=0.018217

--- Numerical Comparison ---
  Tolerance: atol=0.0001, rtol=0.0001
  Max absolute difference:  0.0000004768
  Mean absolute difference: 0.0000000638

  Shape match: [PASS]  PT=(2, 6, 256)  TTSIM=Shape([2, 6, 256])

======================================================================
OVERALL: [PASS] PASS - TTSIM matches PyTorch
======================================================================

```

---
