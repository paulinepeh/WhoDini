import torch
import torch.nn as nn
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo26n-pose.pt") 
net = model.model

print("\n===== INSPECTING MODEL LAYERS =====\n")

# Hook function
def get_hook(name):
    def hook(module, input, output):
        in_shape = input[0].shape if isinstance(input, tuple) else input.shape
        out_shape = output.shape

        print("\n==============================")
        print(f"Layer: {name}")
        print(f"Type : {type(module).__name__}")
        print(f"Input shape : {tuple(in_shape)}")
        print(f"Output shape: {tuple(out_shape)}")

        if isinstance(module, nn.Conv2d):
            print(f"Kernel size: {module.kernel_size}")
            print(f"Stride     : {module.stride}")
            print(f"Padding    : {module.padding}")

        elif isinstance(module, nn.BatchNorm2d):
            print(f"Channels   : {module.num_features}")

        elif isinstance(module, nn.Upsample):
            print(f"Scale factor: {module.scale_factor}")

        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            print(f"Kernel size: {module.kernel_size}")
            print(f"Stride     : {module.stride}")

        print("==============================")

    return hook

for name, layer in net.named_modules():
    if isinstance(layer, (
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.Upsample,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.SiLU,
        nn.ReLU
    )):
        layer.register_forward_hook(get_hook(name))


#  (YOLO default)
x = torch.randn(1, 3, 640, 640)

# Run forward pass
_ = net(x)