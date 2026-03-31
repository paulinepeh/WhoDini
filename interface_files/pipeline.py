import cv2
import hashlib
import numpy as np

class DetectionPipeline:
    def __init__(self, detector, embedder=None, embed_display="hash"):
        self.detector = detector
        self.embedder = embedder
        self.embed_display = embed_display

        self.known_faces = []
        self.next_id = 1
        self.threshold = 15  # tune this

    def match_embedding(self, embedding):
        if len(self.known_faces) == 0:
            self.known_faces.append((self.next_id, embedding))
            self.next_id += 1
            return 0

        distances = []

        for face_id, known_emb in self.known_faces:
            dist = np.linalg.norm(np.array(embedding) - np.array(known_emb))
            distances.append((dist, face_id))

        best_dist, best_id = min(distances)

        if best_dist < self.threshold:
            return best_id
        else:
            self.known_faces.append((self.next_id, embedding))
            self.next_id += 1
            return self.next_id - 1

    def run(self, image):
        detections = self.detector.detect(image)

        results = []

        # 🔥 Run InsightFace ONCE on full image
        faces = []
        if self.embedder:
            faces = self.embedder.app.get(image)

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])

            # --- ADD PADDING ---
            padding_ratio = 0.2  # 20% padding

            w = x2 - x1
            h = y2 - y1

            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)

            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            # Clamp to image boundaries
            img_h, img_w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            embedding = None

            # 🔍 Match face to bbox
            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)

                # Check if face is inside YOLO box
                if fx1 >= x1 and fy1 >= y1 and fx2 <= x2 and fy2 <= y2:
                    embedding = face.embedding.tolist()
                    break

            if embedding is not None:
                # 🔥 Create stable short ID
                embed_id = self.match_embedding(embedding)
                det["embed_id"] = embed_id
            else:
                det["embed_id"] = None

            det["embedding"] = embedding
            results.append(det)

        return results
    



# import cv2

# class DetectionPipeline:
#     def __init__(self, detector, embedder=None):
#         self.detector = detector
#         self.embedder = embedder

#     def run(self, image):
#         detections = self.detector.detect(image)

#         results = []

#         for det in detections:
#             x1, y1, x2, y2 = map(int, det["bbox"])

#             # Clamp
#             h, w = image.shape[:2]
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)

#             crop = image[y1:y2, x1:x2]

#             embedding = None

#             # ✅ Only run InsightFace on "person" class
#             if self.embedder and det["class"] == "person":
#                 embedding = self.embedder.get_embedding(crop)
#                 print(f"Embedding generated: {embedding is not None}")

#             det["embedding"] = embedding
#             results.append(det)

#         return results