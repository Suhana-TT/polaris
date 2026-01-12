import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from workloads.Retinanet import RetinaNet  # Polaris SimNN version

import ttsim.front.functional.op as F

def main():
    cfg = {
        "num_classes": 80,
        "img_size": 608,
        "resnet_depth": 50,
    }

    model = RetinaNet("retinanet_ttsim", cfg)
    model.create_input_tensors()

    cls, reg = model()

    print("Polaris cls shape:", cls.shape)
    print("Polaris reg shape:", reg.shape)

if __name__ == "__main__":
    main()