import cv2
import time

# 0 is “default camera”; try 1 or 2 if you have multiple
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Try a different index (1/2) or check permissions.")

# Optional: set resolution (speeds things up)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

prev = time.time()
frames = 0
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed")
        break

    # (Optional) mirror like a normal selfie view
    frame = cv2.flip(frame, 1)

    # FPS counter
    frames += 1
    now = time.time()
    if now - prev >= 1.0:
        fps = frames / (now - prev)
        prev = now
        frames = 0

    # Draw FPS text
    cv2.putText(frame, f"FPS: {fps:.1f}  (q to quit)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("OpenCV Camera", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
