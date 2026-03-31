import cv2
import time
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from insightface.app import FaceAnalysis

SIM_THRESHOLD = 0.45
FACE_REFRESH_INTERVAL = 2

MOUTH_THRESHOLD = 0.04
NOISE_THRESHOLD = 0.01
SPEAK_FRAMES = 4

SMOOTH_ALPHA = 0.7

print("Loading models...")

model = YOLO("../models/best.pt")

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

whitelist = {}

import os
for file in os.listdir("whitelist"):
    img = cv2.imread(f"whitelist/{file}")
    if img is None:
        continue

    faces = face_app.get(img)
    if len(faces) > 0:
        emb = faces[0].embedding
        whitelist.setdefault("pauline", []).append(emb)

print("Whitelist loaded:", whitelist.keys())

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# ===============================
# MEMORY
# ===============================
track_identity = {}
track_score = {}
track_boxes = {}

prev_mouth = {}
mouth_history = defaultdict(lambda: deque(maxlen=7))

cached_faces = []

frame_idx = 0
prev_time = time.time()

print("Press Q to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    # yolo track
    results = model.track(frame, persist=True, conf=0.25, verbose=False)

    if len(results) == 0 or results[0].boxes is None:
        cv2.imshow("Webcam", frame)
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    track_ids = results[0].boxes.id

    if track_ids is not None:
        track_ids = track_ids.cpu().numpy().astype(int)
    else:
        track_ids = [-1] * len(boxes)

    smoothed = []
    for box, tid in zip(boxes, track_ids):
        x1, y1, x2, y2 = box

        if tid in track_boxes:
            px1, py1, px2, py2 = track_boxes[tid]
            x1 = int(SMOOTH_ALPHA * px1 + (1 - SMOOTH_ALPHA) * x1)
            y1 = int(SMOOTH_ALPHA * py1 + (1 - SMOOTH_ALPHA) * y1)
            x2 = int(SMOOTH_ALPHA * px2 + (1 - SMOOTH_ALPHA) * x2)
            y2 = int(SMOOTH_ALPHA * py2 + (1 - SMOOTH_ALPHA) * y2)

        track_boxes[tid] = [x1, y1, x2, y2]
        smoothed.append((x1, y1, x2, y2, tid))

    #insightface
    if frame_idx % FACE_REFRESH_INTERVAL == 0:
        faces = face_app.get(frame)
        cached_faces = []

        for face in faces:
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            cached_faces.append({
                "bbox": [fx1, fy1, fx2, fy2],
                "emb": face.embedding,
                "kps": face.kps
            })
    for x1, y1, x2, y2, tid in smoothed:

        identity = track_identity.get(tid, "unknown")
        score = track_score.get(tid, 0)

        best_face = None
        best_iou = 0

        for face in cached_faces:
            fx1, fy1, fx2, fy2 = face["bbox"]

            inter = max(0, min(x2, fx2) - max(x1, fx1)) * \
                    max(0, min(y2, fy2) - max(y1, fy1))

            if inter > best_iou:
                best_iou = inter
                best_face = face

        is_speaking = False

        if best_face is not None:
            emb = best_face["emb"]
            kps = best_face["kps"]

            best_sim = -1
            best_name = "unknown"

            for name, refs in whitelist.items():
                for ref in refs:
                    sim = np.dot(emb, ref) / (
                        np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-6
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name

            if best_sim > SIM_THRESHOLD:
                identity = best_name
                score = best_sim
            else:
                identity = "unknown"
                score = best_sim

            track_identity[tid] = identity
            track_score[tid] = score

            #mouth tracking
            if kps is not None:
                mouth = (kps[3] + kps[4]) / 2
                face_width = (x2 - x1)

                if face_width > 0:
                    if tid in prev_mouth:
                        movement = np.linalg.norm(mouth - prev_mouth[tid]) / face_width

                        if movement < NOISE_THRESHOLD:
                            movement = 0

                        speaking_frame = movement > MOUTH_THRESHOLD
                    else:
                        speaking_frame = False

                    prev_mouth[tid] = mouth
                    mouth_history[tid].append(speaking_frame)

                    if sum(mouth_history[tid]) >= SPEAK_FRAMES:
                        is_speaking = True

                 
                    mx, my = int(mouth[0]), int(mouth[1])
                    cv2.circle(frame, (mx, my), 4, (255, 0, 0), -1)


        crop = frame[y1:y2, x1:x2]

        if identity != "pauline" and not is_speaking:
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(crop, (99, 99), 30)

        if identity == "pauline":
            color = (0, 255, 0)
        elif is_speaking:
            color = (255, 255, 0)
        else:
            color = (0, 0, 255)

        label = f"{identity} | {score:.2f}"
        if is_speaking:
            label += " (speaking)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

  
    cv2.putText(
        frame,
        f"FPS: {int(fps_display)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Webcam", frame)

    # EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()