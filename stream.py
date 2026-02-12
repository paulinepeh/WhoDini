import cv2
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame as JPEG
    _, buffer = cv2.imencode(".jpg", frame)
    files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

    try:
        response = requests.post(API_URL, files=files)
        predictions = response.json()["predictions"]

        # Draw predictions
        for pred in predictions:
            x1, y1, x2, y2 = map(int, pred["bbox"])
            label = f"{pred['class']} {pred['confidence']:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    except Exception as e:
        print("API error:", e)

    cv2.imshow("YOLO11s Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
