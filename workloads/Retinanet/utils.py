import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

def conv3x3(name, in_planes, out_planes, stride=1):
    """3x3 convolution with padding (TTSIM)"""
    return F.Conv2d(
        name,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(SimNN.Module):
    expansion = 1

    def __init__(self, objname, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.name = objname

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1",
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1  = F.BatchNorm2d(f"{self.name}_bn1", planes)
        self.relu = F.Relu(f"{self.name}_relu1")

        self.conv2 = F.Conv2d(
            f"{self.name}_conv2",
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = F.BatchNorm2d(f"{self.name}_bn2", planes)

        # downsample should be another SimNN.Module or None
        self.downsample = downsample
        self.stride     = stride

        super().link_op2module()

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)

        return out
    

class Bottleneck(SimNN.Module):
    expansion = 4

    def __init__(self, objname, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.name = objname

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1",
            inplanes,planes,kernel_size=1,stride=1,padding=0,bias=False,
        )
        self.bn1  = F.BatchNorm2d(f"{self.name}_bn1", planes)

        self.conv2 = F.Conv2d(
            f"{self.name}_conv2",planes,planes,kernel_size=3,stride=stride,padding=1,bias=False,
        )
        self.bn2  = F.BatchNorm2d(f"{self.name}_bn2", planes)

        self.conv3 = F.Conv2d(
            f"{self.name}_conv3",planes,planes * Bottleneck.expansion,kernel_size=1,stride=1,padding=0,bias=False,
        )
        self.bn3  = F.BatchNorm2d(f"{self.name}_bn3", planes * Bottleneck.expansion)

        self.relu       = F.Relu(f"{self.name}_relu")
        self.downsample = downsample
        self.stride     = stride

        super().link_op2module()

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out
    

class Downsample(SimNN.Module):
    def __init__(self, objname, inplanes, outplanes, stride):
        super().__init__()
        self.name = objname

        self.conv = F.Conv2d(
            f"{self.name}_conv",
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn = F.BatchNorm2d(f"{self.name}_bn", outplanes)

        super().link_op2module()

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# BBoxTransform and ClipBoxes can be added here if needed

# class BBoxTransform(SimNN.Module):
#     def __init__(self, objname):
#         super().__init__()
#         self.name = objname
#         super().link_op2module()

#     def __call__(self, boxes, deltas):
#         widths  = boxes[:, :, 2] - boxes[:, :, 0]
#         heights = boxes[:, :, 3] - boxes[:, :, 1]
#         ctr_x   = boxes[:, :, 0] + 0.5 * widths
#         ctr_y   = boxes[:, :, 1] + 0.5 * heights

#         dx = deltas[:, :, 0]
#         dy = deltas[:, :, 1]
#         dw = deltas[:, :, 2]
#         dh = deltas[:, :, 3]

#         pred_ctr_x = dx * widths + ctr_x
#         pred_ctr_y = dy * heights + ctr_y
#         pred_w     = F.Exp(f"{self.name}_exp_dw")(dw) * widths
#         pred_h     = F.Exp(f"{self.name}_exp_dh")(dh) * heights

#         pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
#         pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
#         pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
#         pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

#         pred_boxes = F.ConcatX(f"{self.name}_concat", axis=2)(
#             pred_boxes_x1,
#             pred_boxes_y1,
#             pred_boxes_x2,
#             pred_boxes_y2,
#         )
#         return pred_boxes

# class ClipBoxes(SimNN.Module):  
#     def __init__(self, objname):
#         super().__init__()
#         self.name = objname
#         super().link_op2module()

#     def __call__(self, boxes, img):
#         batch_size, _, height, width = F.Shape(f"{self.name}_shape")(img)

#         x1 = F.Max(f"{self.name}_max_x1")(
#             boxes[:, :, 0], F._from_value(f"{self.name}_zero", 0)
#         )
#         y1 = F.Max(f"{self.name}_max_y1")(
#             boxes[:, :, 1], F._from_value(f"{self.name}_zero", 0)
#         )
#         x2 = F.Min(f"{self.name}_min_x2")(
#             boxes[:, :, 2], F._from_value(f"{self.name}_width", width - 1)
#         )
#         y2 = F.Min(f"{self.name}_min_y2")(
#             boxes[:, :, 3], F._from_value(f"{self.name}_height", height - 1)
#         )

#         clipped_boxes = F.ConcatX(f"{self.name}_concat", axis=2)(
#             x1, y1, x2, y2
#         )
#         return clipped_boxes 
