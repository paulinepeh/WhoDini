import ultralytics

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# Load YOLO11s model once on startup
model = YOLO("yolo11s.pt")

@app.get("/")
def read_root():
    return {"status": "YOLO11s Server is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Run Inference
    results = model(image)

    # Parse results
    predictions = []
    for result in results:
        for box in result.boxes:
            predictions.append({
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            })

    return {"predictions": predictions}