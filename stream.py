import cv2
import requests
import time

API_URL = "http://127.0.0.1:8000/predict"

prev_frame_time = 0

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

DISTANCE_THRESHOLD = 30000

while True:
    new_frame_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = new_frame_time

    # Send frame
    _, buffer = cv2.imencode(".jpg", frame)
    files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

    try:
        response = requests.post(API_URL, files=files)
        predictions = response.json().get("predictions", [])

        for pred in predictions:
            x1, y1, x2, y2 = map(int, pred["bbox"])
            
            embed_id = pred.get("embed_id")

            if embed_id:
                label = f"ID:{embed_id}"
            else:
                label = "No Face"

            # embedding = pred.get("embedding")

            # if embedding:
            #     embed_label = "[" + ", ".join(f"{v:.2f}" for v in embedding[:4]) + "]"
            # else:
            #     embed_label = "No Face"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            
            # cv2.putText(frame, label, (x1, y1-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    except Exception as e:
        print("API error:", e)

    cv2.putText(frame, f"FPS:{int(fps)}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import requests
# import numpy as np
# import time  # NEW: For FPS calculation

# API_URL = "http://127.0.0.1:8000/predict"

# # --- FPS GLOBALS ---
# prev_frame_time = 0
# new_frame_time = 0


# cap = cv2.VideoCapture(0) #phone cam: 0   #laptop cam: 1

# if not cap.isOpened():
#     print("Error: Cannot open webcam")
#     exit()

# while True:
#     # Start timer for this frame
#     new_frame_time = time.time()
#     ret, frame = cap.read()
    
#     if not ret:
#         break
    
#     # Calculate FPS
#     # We use 1 / (time_now - time_last_loop)
#     time_diff = new_frame_time - prev_frame_time
#     fps = 1 / time_diff if time_diff > 0 else 0
#     prev_frame_time = new_frame_time

#     # Encode frame as JPEG
#     _, buffer = cv2.imencode(".jpg", frame)
#     files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

#     try:
#         response = requests.post(API_URL, files=files)
#         predictions = response.json()["predictions"]
    
        
#         DISTANCE_THRESHOLD = 30000 #distance trhreshold to blur
#         # Draw predictions 
#         for pred in predictions:
#             x1, y1, x2, y2 = map(int, pred["bbox"])
            
#             obj_id = pred.get("id", "N/A")
#             conf = pred.get("confidence", 0.0)
            
#             # 1. Calculate Area
#             width = x2 - x1
#             height = y2 - y1
#             area = width * height

#             # 2. Logic: Distance-based colors
#             if area < DISTANCE_THRESHOLD:
#                 # FAR / BLURRED
#                 box_color = (0, 0, 255)       # Red Box
#                 pill_color = (255, 255, 255)  # White Label (High Contrast against Red)
#                 text_color = (0, 0, 0)        # Black Text
#                 status = "FAR"
                
#                 # Apply Blur
#                 y1_c, y2_c = max(0, y1), min(frame.shape[0], y2)
#                 x1_c, x2_c = max(0, x1), min(frame.shape[1], x2)
#                 face_roi = frame[y1_c:y2_c, x1_c:x2_c]
#                 if face_roi.size > 0:
#                     frame[y1_c:y2_c, x1_c:x2_c] = cv2.GaussianBlur(face_roi, (99, 99), 50)
#             else:
#                 # CLOSE / CLEAR
#                 box_color = (0, 255, 0)       # Neon Green Box
#                 pill_color = (0, 0, 0)        # Black Label (High Contrast against Green)
#                 text_color = (0, 255, 0)      # Green Text (or White)
#                 status = "CLOSE"

#             # 3. Label String
#             label = f"ID:{obj_id} | {status} | {conf*100:.0f}% | A:{area}"

#             # 4. Text Settings
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 1
#             thickness = 2

#             # 5. Draw the High-Contrast Pill
#             (t_w, t_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
#             # Draw Pill (Background)
#             cv2.rectangle(frame, 
#                           (x1, y1 - t_h - 15), 
#                           (x1 + t_w + 10, y1), 
#                           pill_color, 
#                           -1) 
            
#             # Draw Pill Border (Helps separate the pill from the box)
#             cv2.rectangle(frame, 
#                           (x1, y1 - t_h - 15), 
#                           (x1 + t_w + 10, y1), 
#                           (50, 50, 50), 1)

#             # 6. Draw Text and Bounding Box
#             cv2.putText(frame, label, (x1 + 5, y1 - 10), font, font_scale, text_color, thickness)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3) # Thicker box (3) for visibility
            
            

#     except Exception as e:
#         print("API error:", e)
        
#     # --- DRAW FPS OVERLAY ---
#     fps_text = f"FPS: {int(fps)}"
#     font = cv2.FONT_HERSHEY_SIMPLEX
    
#     # Draw a solid black background pill for FPS in the top-left
#     cv2.rectangle(frame, (10, 10), (120, 45), (0, 0, 0), -1)
#     cv2.putText(frame, fps_text, (20, 35), font, 0.8, (0, 255, 255), 2)

#     cv2.imshow("YOLO Live Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     cv2.imshow("YOLO Live Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# #run code with: python stream.py
