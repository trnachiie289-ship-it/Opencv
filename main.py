import time
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Các cặp điểm nối để vẽ khung xương bàn tay
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)
start = time.monotonic()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break

    rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

    timestamp_ms = int((time.monotonic() - start) * 1000)  # phải tăng dần
    result = landmarker.detect_for_video(image, timestamp_ms)

    h, w, _ = frame.shape
    for hand_landmarks in result.hand_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]

        for a, b in HAND_CONNECTIONS:
            cv.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv.circle(frame, p, 4, (0, 0, 255), -1)

        point = hand_landmarks[0]  # cổ tay
        print(f"{point.x} : {point.y}")

    cv.imshow("frame", cv.flip(frame,1))

    if cv.getWindowProperty("frame", cv.WND_PROP_VISIBLE) < 1:
        break

    if cv.waitKey(1) == ord("q"):
        break

landmarker.close()
cap.release()
cv.destroyAllWindows()
