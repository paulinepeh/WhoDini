import numpy as np
import matplotlib.pyplot as plt

# Load activations
data = np.load("vectors extraction/yolo26_activations.npz")

# Pick a layer
layer_name = data.files[0]  # try different indices later
activation = data[layer_name]

print("Layer:", layer_name)
print("Shape:", activation.shape)

# Take first image, first channel
feature_map = activation[0][0]

plt.imshow(feature_map, cmap='viridis')
plt.title(layer_name)
plt.colorbar()
plt.show()