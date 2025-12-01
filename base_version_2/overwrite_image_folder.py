import os
import cv2

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_LETTER = 'C'          # <-- only this letter
dataset_size = 150

cap = cv2.VideoCapture(1)

WIN_W, WIN_H = 1280, 720
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', WIN_W, WIN_H)

# Make folder for F
class_path = os.path.join(DATA_DIR, TARGET_LETTER)
os.makedirs(class_path, exist_ok=True)

print(f'Collecting data for Letter: {TARGET_LETTER}')

# -------- preview loop --------
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame read failed.")
        continue

    frame_resized = cv2.resize(frame, (WIN_W, WIN_H), interpolation=cv2.INTER_AREA)

    cv2.putText(frame_resized, f"Ready for Letter {TARGET_LETTER}?",
                (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_resized, "Press S to start",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame_resized)

    if cv2.waitKey(25) & 0xFF == ord('s'):
        break

# -------- capture loop --------
counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame read failed during capture.")
        continue

    frame_resized = cv2.resize(frame, (WIN_W, WIN_H), interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame_resized)
    cv2.waitKey(25)

    img_path = os.path.join(class_path, f"{counter}.jpg")
    cv2.imwrite(img_path, frame)   # or frame_resized if you prefer
    counter += 1

cap.release()
cv2.destroyAllWindows()
