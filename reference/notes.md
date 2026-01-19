#!/usr/bin/env python3

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

    x0 = model.backbone.conv1(x)
    x0 = model.backbone.bn1(x0)
    x0 = model.backbone.relu(x0)
    x0 = model.backbone.maxpool(x0)

    x1 = x0
    for m in model.backbone.layer1:
        x1 = m(x1)

    x2 = x1
    for m in model.backbone.layer2:
        x2 = m(x2)

    x3 = x2
    for m in model.backbone.layer3:
        x3 = m(x3)

    x4 = x3
    for m in model.backbone.layer4:
        x4 = m(x4)

    features = model.fpn([x2, x3, x4])

    reg_list = [model.regressionModel(f) for f in features]
    cls_list = [model.classificationModel(f) for f in features]

    reg_concat = F.ConcatX("retinanet_ttsim_reg_concat", axis=1)(*reg_list)
    cls_concat = F.ConcatX("retinanet_ttsim_cls_concat", axis=1)(*cls_list)

    anchors_np = model.anchors(img_size, img_size) 
    
    transformed_anchors = model.regressBoxes(anchors_np, reg_concat)

    final_bbox_coords = model.clipBoxes(transformed_anchors, x)


    print("Polaris cls shape:      ", cls_concat.shape)       # [1, 69354, 80]
    print("Polaris reg shape:      ", reg_concat.shape)       # [1, 69354, 4]
    print("Final BBox Coords shape:", final_bbox_coords.shape)  # [1, 69354, 4]
    
if __name__ == "__main__":
    main()