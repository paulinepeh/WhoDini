import cv2

class DetectionPipeline:
    def __init__(self, detector, embedder=None):
        self.detector = detector
        self.embedder = embedder

    def run(self, image):
        detections = self.detector.detect(image)

        results = []

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])

            # Clamp
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = image[y1:y2, x1:x2]

            embedding = None

            # ✅ Only run InsightFace on "person" class
            if self.embedder and det["class"] == "person":
                embedding = self.embedder.get_embedding(crop)
                print(f"Embedding generated: {embedding is not None}")

            det["embedding"] = embedding
            results.append(det)

        return results