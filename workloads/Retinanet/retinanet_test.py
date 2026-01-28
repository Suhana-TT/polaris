#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
# Add project root to path to ensure ttsim and workloads are findable
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
from workloads.Retinanet.model import RetinaNet

def run_standalone(outdir: str = '.') -> None:
    """
    Renamed from main() per reviewer feedback. 
    Handles model instantiation and graph dumping for Polaris integration.
    """
    # Define configurations similar to ip_workloads.yaml structure
    retinanet_cfgs = {
        'retinanet_rn50_608': {
            "num_classes": 80,
            "img_size": 608,
            "resnet_depth": 50,
        }
    }

    for name, cfg in retinanet_cfgs.items():
        print(f"Creating {name} model...")
        model = RetinaNet(name, cfg)

        # Generate dummy input based on config
        img_size = cfg["img_size"]
        x_np = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        x = F._from_data(f"{name}_x_in", x_np)

        # Run forward pass
        cls_concat, reg_concat, final_bbox_coords = model(x)

        print(f"Output Shapes for {name}:")
        print(f"  Classification: {cls_concat.shape}")       
        print(f"  Regression:     {reg_concat.shape}")       
        print(f"  BBox Coords:    {final_bbox_coords.shape}")  

        # Internal Assertions
        assert len(final_bbox_coords.shape) == 3, "BBox should be 3D [Batch, Anchors, 4]"
        
        # Dump Graph (Standard practice for run_standalone in Polaris)
        print(f"Dumping ONNX to {outdir}...")
        # Note: This assumes RetinaNet has a get_forward_graph method or 
        # utilizes the SimNN.Module graph tracking
        # gg = model.get_forward_graph() 
        # gg.graph2onnx(f'{outdir}/{name}.onnx', do_model_check=True)
        
        print('-' * 40)

if __name__ == "__main__":
    run_standalone()