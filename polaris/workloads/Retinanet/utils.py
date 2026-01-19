import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np

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

class BBoxTransform(SimNN.Module):
    def __init__(self, objname):
        super().__init__()
        self.name = objname
        self.std_values = [0.1, 0.1, 0.2, 0.2]
        self.mean_values = [0.0, 0.0, 0.0, 0.0]

    def __call__(self, boxes, deltas):
        deltas.link_module = self
        if isinstance(boxes, np.ndarray):
            boxes = F._from_data(f"{self.name}_anchors", boxes)
        boxes.link_module = self

        half = F._from_data(f"{self.name}_half", np.array([0.5], dtype=np.float32))
        
        s0 = F._from_data(f"{self.name}_s0", np.array([self.std_values[0]], dtype=np.float32))
        s1 = F._from_data(f"{self.name}_s1", np.array([self.std_values[1]], dtype=np.float32))
        s2 = F._from_data(f"{self.name}_s2", np.array([self.std_values[2]], dtype=np.float32))
        s3 = F._from_data(f"{self.name}_s3", np.array([self.std_values[3]], dtype=np.float32))
        
        m0 = F._from_data(f"{self.name}_m0", np.array([self.mean_values[0]], dtype=np.float32))
        m1 = F._from_data(f"{self.name}_m1", np.array([self.mean_values[1]], dtype=np.float32))
        m2 = F._from_data(f"{self.name}_m2", np.array([self.mean_values[2]], dtype=np.float32))
        m3 = F._from_data(f"{self.name}_m3", np.array([self.mean_values[3]], dtype=np.float32))

    
        d_list = F.Split(f"{self.name}_d_split", axis=2, count=4)(deltas)
        dx, dy, dw, dh = d_list[0], d_list[1], d_list[2], d_list[3]
                
        for d in [dx, dy, dw, dh]: d.link_module = self

        b_list = F.Split(f"{self.name}_b_split", axis=2, count=4)(boxes)
        x1_a, y1_a, x2_a, y2_a = b_list[0], b_list[1], b_list[2], b_list[3]
        
        for b in [x1_a, y1_a, x2_a, y2_a]: b.link_module = self

        widths  = x2_a - x1_a
        heights = y2_a - y1_a
        ctr_x   = x1_a + (widths * half)
        ctr_y   = y1_a + (heights * half)

        dx_n = dx * s0 + m0
        dy_n = dy * s1 + m1
        dw_n = dw * s2 + m2
        dh_n = dh * s3 + m3

        p_ctr_x = ctr_x + (dx_n * widths)
        p_ctr_y = ctr_y + (dy_n * heights)
        p_w     = F.Exp(f"{self.name}_exp_w")(dw_n) * widths
        p_h     = F.Exp(f"{self.name}_exp_h")(dh_n) * heights

        px1 = p_ctr_x - (p_w * half)
        py1 = p_ctr_y - (p_h * half)
        px2 = p_ctr_x + (p_w * half)
        py2 = p_ctr_y + (p_h * half)

        return F.ConcatX(f"{self.name}_concat", axis=2)(px1, py1, px2, py2)

class ClipBoxes(SimNN.Module):
    def __init__(self, objname):
        super().__init__()
        self.name = objname

    def __call__(self, boxes, img):
        boxes.link_module = self
        
        b_list = F.Split(f"{self.name}_b_split_clip", axis=2, count=4)(boxes)
        bx1, by1, bx2, by2 = b_list[0], b_list[1], b_list[2], b_list[3]

        for b in [bx1, by1, bx2, by2]: b.link_module = self
        
        px1 = F.Clip(f"{self.name}_clip_x1", min=0.0, max=607.0)(bx1)
        py1 = F.Clip(f"{self.name}_clip_y1", min=0.0, max=607.0)(by1)
        px2 = F.Clip(f"{self.name}_clip_x2", min=0.0, max=607.0)(bx2)
        py2 = F.Clip(f"{self.name}_clip_y2", min=0.0, max=607.0)(by2)

        return F.ConcatX(f"{self.name}_concat_clip", axis=2)(px1, py1, px2, py2)
    

def nms(boxes, scores, threshold=0.5):
    
    if len(boxes) == 0: return []
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep