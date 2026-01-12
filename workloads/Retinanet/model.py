#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

""" Parts of the Retinanet model """
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


from workloads.Retinanet.utils import BasicBlock, Bottleneck, Downsample

class PyramidFeatures(SimNN.Module):
    def __init__(self, objname, C3_size, C4_size, C5_size, feature_size=256):
        super().__init__()
        self.name = objname

        # P5: upsample C5 to get P5
        self.P5_1 = F.Conv2d(
            f"{self.name}_P5_1",C5_size,feature_size,kernel_size=1,stride=1,padding=0,bias=False,
        )
        # self.P5_upsampled = F.Upsample(
        #     f"{self.name}_P5_upsample",scale_factor=2,mode="nearest",
        # )
        self.P5_2 = F.Conv2d(
            f"{self.name}_P5_2",feature_size,feature_size,kernel_size=3,stride=1,padding=1,bias=False,
        )

        # P4: add P5 (upsampled) elementwise to C4
        self.P4_1 = F.Conv2d(
            f"{self.name}_P4_1",C4_size,feature_size,kernel_size=1,stride=1,padding=0,bias=False,
        )
        # self.P4_upsampled = F.Upsample(
        #     f"{self.name}_P4_upsample",scale_factor=2,mode="nearest",
        # )
        self.P4_2 = F.Conv2d(
            f"{self.name}_P4_2",feature_size,feature_size,kernel_size=3,stride=1,padding=1,bias=False,
        )

        # P3: add P4 (upsampled) elementwise to C3
        self.P3_1 = F.Conv2d(
            f"{self.name}_P3_1",C3_size,feature_size,kernel_size=1,stride=1,padding=0,bias=False,
        )
        self.P3_2 = F.Conv2d(
            f"{self.name}_P3_2",feature_size,feature_size,kernel_size=3,stride=1,padding=1,bias=False,
        )

        # P6: 3×3 stride‑2 conv on C5
        self.P6 = F.Conv2d(
            f"{self.name}_P6",C5_size,feature_size,kernel_size=3,stride=2,padding=1,bias=False,
        )

        # P7: ReLU then 3×3 stride‑2 conv on P6
        self.P7_1 = F.Relu(f"{self.name}_P7_relu")
        self.P7_2 = F.Conv2d(
            f"{self.name}_P7_2",feature_size,feature_size,kernel_size=3,stride=2,padding=1,bias=False,
        )
        super().link_op2module()

    def __call__(self, inputs):
        C3, C4, C5 = inputs

    # P5
        P5_x = self.P5_1(C5)
    # use SimTensor.interpolate instead of F.Upsample
        P5_upsampled_x = P5_x.interpolate(scale_factor=2.0, mode="nearest")  # type: ignore[attr-defined]
        self._tensors[P5_upsampled_x.name] = P5_upsampled_x
        P5_x = self.P5_2(P5_x)

    # P4
        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x
        P4_upsampled_x = P4_x.interpolate(scale_factor=2.0, mode="nearest")  # type: ignore[attr-defined]
        self._tensors[P4_upsampled_x.name] = P4_upsampled_x
        P4_x = self.P4_2(P4_x)

    # P3
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

    # P6, P7 same as before
        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(SimNN.Module):
    def __init__(self, objname, num_features_in, num_anchors=9, feature_size=256):
        super().__init__()
        self.name = objname
        self.num_anchors = num_anchors

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1",num_features_in,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act1 = F.Relu(f"{self.name}_relu1")

        self.conv2 = F.Conv2d(
            f"{self.name}_conv2",feature_size,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act2 = F.Relu(f"{self.name}_relu2")

        self.conv3 = F.Conv2d(
            f"{self.name}_conv3",feature_size,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act3 = F.Relu(f"{self.name}_relu3")

        self.conv4 = F.Conv2d(
            f"{self.name}_conv4",feature_size,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act4 = F.Relu(f"{self.name}_relu4")

        self.output = F.Conv2d(
            f"{self.name}_out",feature_size,num_anchors * 4,kernel_size=3,padding=1,bias=False,
        )

        super().link_op2module()

    def __call__(self, x):
        out = self.conv1(x); out = self.act1(out)
        out = self.conv2(out); out = self.act2(out)
        out = self.conv3(out); out = self.act3(out)
        out = self.conv4(out); out = self.act4(out)
        out = self.output(out)

        # out is B x (A*4) x H x W → B x H x W x (A*4)
        out = out.permute([0, 2, 3, 1])
        b = out.shape[0]

        return out.view(b, -1, 4)
    


class ClassificationModel(SimNN.Module):
    def __init__(self, objname, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super().__init__()
        self.name        = objname
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1",num_features_in,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act1 = F.Relu(f"{self.name}_relu1")

        self.conv2 = F.Conv2d(
            f"{self.name}_conv2",feature_size,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act2 = F.Relu(f"{self.name}_relu2")

        self.conv3 = F.Conv2d(
            f"{self.name}_conv3",feature_size,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act3 = F.Relu(f"{self.name}_relu3")

        self.conv4 = F.Conv2d(
            f"{self.name}_conv4",feature_size,feature_size,kernel_size=3,padding=1,bias=False,
        )
        self.act4 = F.Relu(f"{self.name}_relu4")

        self.output = F.Conv2d(
            f"{self.name}_out",feature_size,num_anchors * num_classes,kernel_size=3,padding=1,bias=False,
        )
        self.output_act = F.Sigmoid(f"{self.name}_sigmoid")

        super().link_op2module()

    def __call__(self, x):
        out = self.conv1(x); out = self.act1(out)
        out = self.conv2(out); out = self.act2(out)
        out = self.conv3(out); out = self.act3(out)
        out = self.conv4(out); out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # B x (A*C) x H x W → B x H x W x (A*C)
        out1 = out.permute([0, 2, 3, 1])
        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.view(batch_size, -1, self.num_classes)
    
class ResNet(SimNN.Module):
    def __init__(self, objname, block, layers):
        super().__init__()
        self.name     = objname
        self.inplanes = 64

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1",in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False,
        )
        self.bn1   = F.BatchNorm2d(f"{self.name}_bn1", 64)
        self.relu  = F.Relu(f"{self.name}_relu")
        self.maxpool = F.MaxPool2d(
            f"{self.name}_maxpool",kernel_size=3,stride=2,padding=1,
        )

        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        super().link_op2module()
 
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []

        outplanes = planes * block.expansion

    # downsample if stride changes or channels change
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = Downsample(
                f"{self.name}_down_{planes}",self.inplanes,outplanes,stride,
           )

    # first block in this layer
        layers.append(block(
            f"{self.name}_layer{planes}_0",
            self.inplanes, planes, stride=stride,downsample=downsample,
        ))
        self.inplanes = outplanes

    # remaining blocks
        for i in range(1, blocks):
            layers.append(block(
                f"{self.name}_layer{planes}_{i}",self.inplanes, planes, stride=1,downsample=None,
            ))

        mlist = SimNN.ModuleList(layers)
        for m in mlist:
            self._submodules[m.name] = m
        return mlist

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

    # layer1 → C2 (we won't return this)
        out = x
        for m in self.layer1:
           out = m(out)
        C2 = out

    # layer2 → C3
        for m in self.layer2:
            out = m(out)
        C3 = out

    # layer3 → C4
        for m in self.layer3:
            out = m(out)
        C4 = out

    # layer4 → C5
        for m in self.layer4:
            out = m(out)
        C5 = out

    # Return C3, C4, C5 as in FPN paper
        return C3, C4, C5
    
class RetinaNet(SimNN.Module):
    def __init__(self, objname, cfg):
        super().__init__()
        self.name        = objname
        self.num_classes = cfg["num_classes"]
        self.img_size    = cfg["img_size"]
        depth            = cfg.get("resnet_depth", 50)

        if depth == 18:
            block, layers = BasicBlock, [2,2,2,2]
            C3_size, C4_size, C5_size = 128, 256, 512
        elif depth == 50:
            block, layers = Bottleneck, [3,4,6,3]
            C3_size, C4_size, C5_size = 512, 1024, 2048
        # etc

        self.backbone = ResNet(f"{self.name}_backbone", block, layers)
        self.fpn      = PyramidFeatures(f"{self.name}_fpn", C3_size, C4_size, C5_size)
        self.reg_head = RegressionModel(f"{self.name}_reg", 256)
        self.cls_head = ClassificationModel(f"{self.name}_cls", 256, num_classes=self.num_classes)

        super().link_op2module()

    def create_input_tensors(self):
        self.input_tensors = {
            "x_in": F._from_shape("x_in", [1, 3, self.img_size, self.img_size]),
        }

    def __call__(self, x=None):
        x = self.input_tensors["x_in"] if x is None else x
        C3, C4, C5 = self.backbone.forward(x)
        features   = self.fpn([C3, C4, C5])

        reg = F.ConcatX(f"{self.name}_reg_concat", axis=1)(
            *[self.reg_head(f) for f in features]
        )
        cls = F.ConcatX(f"{self.name}_cls_concat", axis=1)(
            *[self.cls_head(f) for f in features]
        )
        return cls, reg

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    
def resnet18_backbone(objname):
        # ResNet‑18: BasicBlock, [2,2,2,2]
        return ResNet(objname, BasicBlock, [2, 2, 2, 2])

def resnet34_backbone(objname):
        # ResNet‑34: BasicBlock, [3,4,6,3]
        return ResNet(objname, BasicBlock, [3, 4, 6, 3])

def resnet50_backbone(objname):
        # ResNet‑50: Bottleneck, [3,4,6,3]
        return ResNet(objname, Bottleneck, [3, 4, 6, 3])

def resnet101_backbone(objname):
        # ResNet‑101: Bottleneck, [3,4,23,3]
        return ResNet(objname, Bottleneck, [3, 4, 23, 3])

def resnet152_backbone(objname):
        # ResNet‑152: Bottleneck, [3,8,36,3]
        return ResNet(objname, Bottleneck, [3, 8, 36, 3])
    


