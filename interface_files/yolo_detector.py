from ultralytics import YOLO
from .interfaces import Detector

class YOLODetector(Detector):
    def __init__(self, model_path, tracker="bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.tracker = tracker

    def detect(self, image):
        results = self.model.track(image, persist=True, tracker=self.tracker)

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                track_id = int(box.id[0]) if box.id is not None else None

                detections.append({
                    "id": track_id,
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                })

        return detections