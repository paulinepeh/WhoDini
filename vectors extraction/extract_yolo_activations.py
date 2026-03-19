from ultralytics import YOLO
import torch
import numpy as np
import torch.nn as nn

#  Load model
model = YOLO("yolo26n-pose.pt") 
net = model.model
net.eval()

#  Extract weights
weights_np = {}

for name, param in net.state_dict().items():
    weights_np[name] = param.cpu().numpy()

np.savez("yolo26_weights.npz", **weights_np)
print(" Weights saved!")

#  Capture activations
activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

# Only activation layers
for name, layer in net.named_modules():
    if isinstance(layer, (nn.ReLU, nn.SiLU, nn.LeakyReLU)):
        layer.register_forward_hook(get_hook(name))

#  Run forward pass
dummy_input = torch.randn(1, 3, 640, 640)
_ = net(dummy_input)

#  Save activations
np.savez("yolo26_activations.npz", **activations)
print(" Activations saved")