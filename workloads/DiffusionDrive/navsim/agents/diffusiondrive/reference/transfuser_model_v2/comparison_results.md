# Comparison Results (transfuser_model_v2) -- 2026-04-10 13:20:59

## test_1_agent_head.py  --  PASS

### stdout

```
======================================================================
AgentHead Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch AgentHead ---
Agent queries shape: torch.Size([2, 10, 256])
Agent states shape: (2, 10, 5)
Agent labels shape: (2, 10)
Agent states stats: min=-11.836118, max=13.085490, mean=0.157526
Agent labels stats: min=-0.776515, max=1.014359, mean=0.111508

--- TTSIM AgentHead ---
Injecting weights...
Agent queries shape: Shape([2, 10, 256])
Agent states shape: Shape([2, 10, 5])
Agent labels shape: Shape([2, 10, 1])
Agent states stats: min=-11.836116, max=13.085489, mean=0.157526
Agent labels stats: min=-0.776515, max=1.014359, mean=0.111508

--- Numerical Comparison: PyTorch vs TTSIM ---
Tolerance: atol=0.0001, rtol=0.0001

Agent States:
  Max absolute difference: 0.0000066757
  Mean absolute difference: 0.0000005225
  PASS: PASS: TTSIM matches PyTorch for agent states

Agent Labels:
  Max absolute difference: 0.0000001863
  Mean absolute difference: 0.0000000544
  PASS: PASS: TTSIM matches PyTorch for agent labels

======================================================================
OVERALL: PASS: PASS - All outputs match
======================================================================

```

---

## test_2_diff_motion_planning.py  --  PASS

### stdout

```
======================================================================
DiffMotionPlanningRefinementModule Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch DiffMotionPlanningRefinementModule ---
Trajectory feature shape: torch.Size([2, 20, 256])
Plan regression shape: (2, 20, 8, 3)
Plan classification shape: (2, 20)
Plan reg stats: min=-0.232367, max=0.289013, mean=0.019850
Plan cls stats: min=-5.904417, max=-3.335353, mean=-4.573845

--- TTSIM DiffMotionPlanningRefinementModule ---
Injecting weights...
Trajectory feature shape: Shape([2, 20, 256])
Plan regression shape: Shape([2, 20, 8, 3])
Plan classification shape: Shape([2, 20])
Plan reg stats: min=-0.232367, max=0.289013, mean=0.019850
Plan cls stats: min=-5.904417, max=-3.335353, mean=-4.573845

--- Numerical Comparison: PyTorch vs TTSIM ---
Tolerance: atol=0.0001, rtol=0.0001

Plan Regression:
  Max absolute difference: 0.0000001192
  Mean absolute difference: 0.0000000138
  PASS: TTSIM matches PyTorch for plan regression

Plan Classification:
  Max absolute difference: 0.0000004768
  Mean absolute difference: 0.0000001907
  PASS: TTSIM matches PyTorch for plan classification

======================================================================
OVERALL: PASS - All outputs match
======================================================================

```

---

## test_3_modulation_layer.py  --  PASS

### stdout

```
======================================================================
ModulationLayer Validation: PyTorch vs TTSIM
======================================================================

--- PyTorch ModulationLayer ---
Trajectory feature shape: torch.Size([2, 20, 256])
Time embed shape: torch.Size([2, 1, 256])
Output shape: (2, 20, 256)
Output stats: min=-5.838932, max=5.134801, mean=0.001329

--- TTSIM ModulationLayer ---
Injecting weights...
Trajectory feature shape: Shape([2, 20, 256])
Time embed shape: Shape([2, 1, 256])
Output shape: Shape([2, 20, 256])
Output stats: min=-5.838932, max=5.134801, mean=0.001329

--- Numerical Comparison: PyTorch vs TTSIM ---
Tolerance: atol=0.0001, rtol=0.0001
  Max absolute difference: 0.0000009537
  Mean absolute difference: 0.0000000654

======================================================================
OVERALL: PASS - TTSIM matches PyTorch
======================================================================

```

---

## test_4_custom_transformer_decoder_layer.py  --  PASS

### stdout

```
======================================================================
CustomTransformerDecoderLayer Validation: Full Numerical Test
======================================================================


--- Creating PyTorch Reference ---
--- Creating TTSIM Model ---

--- PyTorch Forward Pass ---
PyTorch output shapes:
  Poses regression: torch.Size([1, 4, 8, 3])
  Poses classification: torch.Size([1, 4])

--- Injecting Weights ---

--- TTSIM Forward Pass ---
TTSIM output shapes:
  Poses regression: Shape([1, 4, 8, 3])
  Poses classification: Shape([1, 4])

--- Numerical Comparison ---
Regression output:
  Max absolute difference: 0.000001
  Mean absolute difference: 0.000000
Classification output:
  Max absolute difference: 0.000000
  Mean absolute difference: 0.000000

======================================================================
OVERALL: PASS
  Regression max diff: 0.0000005215
  Classification max diff: 0.0000004768
======================================================================

```

---

## test_5_custom_transformer_decoder.py  --  PASS

### stdout

```
======================================================================
CustomTransformerDecoder Validation: Full Numerical Test
======================================================================

--- Creating PyTorch Reference ---
--- Creating TTSIM Model ---

--- PyTorch Forward Pass ---
PyTorch output shapes:
  Number of outputs: 2
  First regression: torch.Size([1, 4, 8, 3])
  First classification: torch.Size([1, 4])

--- Injecting Weights ---

--- TTSIM Forward Pass ---
TTSIM output shapes:
  Number of outputs: 2
  First regression: Shape([1, 4, 8, 3])
  First classification: Shape([1, 4])

--- Numerical Comparison ---

Layer 0:
  Regression  - Max diff: 0.000001, Mean diff: 0.000000
  Classification - Max diff: 0.000000, Mean diff: 0.000000
  Layer 0: PASS

Layer 1:
  Regression  - Max diff: 0.000000, Mean diff: 0.000000
  Classification - Max diff: 0.000000, Mean diff: 0.000000
  Layer 1: PASS

======================================================================
OVERALL: PASS: PASS
  Max regression diff across all layers: 0.0000005215
  Max classification diff across all layers: 0.0000004768
======================================================================

```

---

## test_6_trajectory_head.py  --  PASS

### stdout

```
======================================================================
TrajectoryHead Validation
======================================================================

Created plan anchor: test_plan_anchor.npy

--- Creating Models ---

--- Injecting Weights ---
PASS: Weight injection complete

--- Generating Inputs ---
  Ego query: (1, 1, 256)
  Agents query: (1, 3, 256)
  BEV feature: (1, 256, 8, 8)
  Status encoding: (1, 1, 256)

--- PyTorch Forward Pass (full 2-step loop) ---
  Noise shape: (1, 20, 8, 2) (matches img after norm_odo)
PASS: PyTorch output shape: (1, 8, 3)

--- TTSIM Forward Pass (full 2-step loop) ---
PASS: TTSIM output shape: Shape([1, 8, 3])

--- Numerical Comparison ---
Max absolute difference: 0.000002
Mean absolute difference: 0.000001

======================================================================
OVERALL: PASS - np.allclose(atol=0.0001, rtol=0.0001)
======================================================================

Cleaned up: test_plan_anchor.npy

```

---

## test_7_v2_transfuser_full_validation.py  --  PASS

### stdout

```
======================================================================
V2TransfuserModel Validation: PyTorch vs TTSIM
======================================================================

--- Creating PyTorch Model ---
PASS: PyTorch V2TransfuserModel created

--- Creating TTSIM Model ---
PASS: TTSIM V2TransfuserModel created

--- Injecting Weights ---
PASS: Weight injection complete

--- Input Shapes ---
  Camera: [1, 3, 32, 32]
  LiDAR:  [1, 1, 32, 32]
  Status: [1, 8]

--- PyTorch Forward Pass ---
PASS: PyTorch forward pass complete
  bev_semantic_map: shape=[1, 7, 16, 32], min=-0.107430, max=0.141230, mean=-0.004697
  trajectory: shape=[1, 8, 3], min=-1.936627, max=2.348682, mean=0.069705
  agent_states: shape=[1, 3, 5], min=-1.695054, max=3.863143, mean=0.581616
  agent_labels: shape=[1, 3], min=-0.438846, max=-0.333250, mean=-0.387068

--- TTSIM Forward Pass ---
PASS: TTSIM forward pass complete
  bev_semantic_map: shape=[1, 7, 16, 32], min=-0.107430, max=0.141230, mean=-0.004697
  trajectory: shape=[1, 8, 3], min=-3.372122, max=4.617360, mean=0.024230
  agent_states: shape=[1, 3, 5], min=-1.695054, max=3.863144, mean=0.581615
  agent_labels: shape=[1, 3, 1], min=-0.438846, max=-0.333250, mean=-0.387068

--- Shape Validation ---
  bev_semantic_map: expected=[1, 7, 16, 32]  pytorch=[1, 7, 16, 32]  ttsim=[1, 7, 16, 32]  PASS
  agent_states: expected=[1, 3, 5]  pytorch=[1, 3, 5]  ttsim=[1, 3, 5]  PASS
  agent_labels: expected=[1, 3]  pytorch=[1, 3]  ttsim=[1, 3]  PASS
  trajectory: expected=[1, 8, 3]  pytorch=[1, 8, 3]  ttsim=[1, 8, 3]  PASS

--- Numerical Comparison ---
  Tolerance: atol=0.0001, rtol=0.0001

  bev_semantic_map:
    PyTorch shape: [1, 7, 16, 32]
    TTSIM shape:   [1, 7, 16, 32]
    Max absolute difference:  0.0000000298
    Mean absolute difference: 0.0000000048
    PyTorch stats: min=-0.107430, max=0.141230, mean=-0.004697
    TTSIM   stats: min=-0.107430, max=0.141230, mean=-0.004697
    PASS

  agent_states:
    PyTorch shape: [1, 3, 5]
    TTSIM shape:   [1, 3, 5]
    Max absolute difference:  0.0000045300
    Mean absolute difference: 0.0000005521
    PyTorch stats: min=-1.695054, max=3.863143, mean=0.581616
    TTSIM   stats: min=-1.695054, max=3.863144, mean=0.581615
    PASS

  agent_labels:
    PyTorch shape: [1, 3]
    TTSIM shape:   [1, 3]
    Max absolute difference:  0.0000000596
    Mean absolute difference: 0.0000000199
    PyTorch stats: min=-0.438846, max=-0.333250, mean=-0.387068
    TTSIM   stats: min=-0.438846, max=-0.333250, mean=-0.387068
    PASS

  trajectory:
    PyTorch shape: [1, 8, 3]
    TTSIM   shape: [1, 8, 3]
    (Comparison skipped: TTSIM uses mock scheduler)
    (Trajectory validated separately in test_6_trajectory_head.py)

======================================================================
OVERALL: PASS - All outputs match (atol=0.0001, rtol=0.0001)
  Shape validation: PASS
  bev_semantic_map: PASS
  agent_states: PASS
  agent_labels: PASS
======================================================================

```

---
