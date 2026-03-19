import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

def preprocess_image(image_path, input_shape):
    """
    image_path: path to your image file
    input_shape: (N, C, H, W) from model
    """

    _, c, h, w = input_shape
    
    with Image.open('t1.jpg') as img:
        w, h = img.size

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize to model input size
    img = cv2.resize(img, (w, h))

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize (adjust if needed)
    img = img.astype(np.float32) / 255.0

    # HWC → CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# Load ONNX model
session = ort.InferenceSession("det_500m_intermediate.onnx")

# Get model input info
input_info = session.get_inputs()[0]
input_name = input_info.name
input_shape = input_info.shape  # e.g. [1, 3, 640, 640]

# Load and preprocess image
image_path = "t1.jpg"  # <-- put your filename here
input_data = preprocess_image(image_path, input_shape)

# Run inference
outputs = session.run(None, {input_name: input_data})

print("Total outputs:", len(outputs))

# Print output shapes
for i, out in enumerate(outputs):
    print(f"Output {i}: shape = {out.shape}")