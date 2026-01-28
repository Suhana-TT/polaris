#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import ttsim.front.functional.op as F
from workloads.Retinanet.model import RetinaNet

def main():
    num_classes = 80
    img_size    = 608

    cfg = {
        "num_classes": num_classes,
        "img_size": img_size,
        "resnet_depth": 50,  
    }

    model = RetinaNet("retinanet_ttsim", cfg)

    x_np = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    x    = F._from_data("x_in_test", x_np)

    # Calling the model once returns the three tensors
    cls_concat, reg_concat, final_bbox_coords = model(x)

    print("Retinanet cls shape:      ", cls_concat.shape)       
    print("Retinanet reg shape:      ", reg_concat.shape)       
    print("Final BBox Coords shape:", final_bbox_coords.shape)  
    
    assert list(final_bbox_coords.shape) == [1, 69354, 4], f"Expected bbox shape [1, 69354, 4], got {final_bbox_coords.shape}"
    assert list(cls_concat.shape) == [1, 69354, 80], f"Expected cls shape [1, 69354, 80], got {cls_concat.shape}"
    assert list(reg_concat.shape) == [1, 69354, 4], f"Expected reg shape [1, 69354, 4], got {reg_concat.shape}"
    
    print("All assertions passed!")
    
if __name__ == "__main__":
    main()