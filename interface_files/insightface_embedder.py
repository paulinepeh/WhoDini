import insightface
import cv2
from .interfaces import Embedder

class InsightFaceEmbedder(Embedder):
    def __init__(self, ctx_id=0):
        self.app = insightface.app.FaceAnalysis()
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def get_embedding(self, crop):
        if crop is None or crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        faces = self.app.get(crop_rgb)

        if len(faces) > 0:
            return faces[0].embedding.tolist()

        return None

# from insightface.app import FaceAnalysis
# import cv2
# from .interfaces import Embedder

# class InsightFaceEmbedder(Embedder):
#     def __init__(self, ctx_id=0):
#         self.app = FaceAnalysis(name="buffalo_sc")
#         self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

#     def get_embedding(self, crop):
#         if crop is None or crop.size == 0:
#             return None

#         crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

#         faces = self.app.get(crop_rgb)

#         if len(faces) > 0:
#             return faces[0].embedding.tolist()

#         return None