# retinanet/test.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torchvision.ops import nms 
from retinanet.model import resnet50 

def main():
    num_classes = 80
    img_size = 608

    model = resnet50(num_classes=num_classes, pretrained=False)
    model.eval()

    x = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        x0 = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        x1 = model.layer1(x0)
        x2 = model.layer2(x1)
        x3 = model.layer3(x2)
        x4 = model.layer4(x3)

        features = model.fpn([x2, x3, x4])

        reg = torch.cat([model.regressionModel(f) for f in features], dim=1)
        cls = torch.cat([model.classificationModel(f) for f in features], dim=1)

        anchors = model.anchors(x) 

        transformed_anchors = model.regressBoxes(anchors, reg)

        final_bbox_coords = model.clipBoxes(transformed_anchors, x)

    print("PyTorch cls shape:", cls.shape)              # Raw scores
    print("PyTorch reg shape:", reg.shape)              # Raw offsets
    print("Final BBox Coords shape:", final_bbox_coords.shape) # Actual Pixels [1, NumAnchors, 4]

if __name__ == "__main__":
    main()