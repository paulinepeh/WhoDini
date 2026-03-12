import ultralytics

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()
# Load YOLO11s model once on startup
model = YOLO("models/yolo26n_trained_by_nibras.pt") #set model

@app.get("/")
def read_root():
    return {"status": "YOLO11s Server is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = model.track(image, persist=True, tracker="bytetrack.yaml")

    predictions = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Get the ID. We use .get() or a check because 
                # sometimes a box is detected but not yet tracked.
                track_id = int(box.id[0]) if box.id is not None else None
                
                predictions.append({
                    "id": track_id,
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

    return {"predictions": predictions}
    


# uvicorn main:app --reload