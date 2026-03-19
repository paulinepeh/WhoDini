import numpy as np

data = np.load("yolo26_weights.npz")

print("Total layers:", len(data.files))
print("\n--- First 20 layers ---")

for name in data.files[:20]:
    print(name, data[name].shape)

    