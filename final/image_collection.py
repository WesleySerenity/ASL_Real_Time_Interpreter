import os
import cv2
import string


DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

class_labels = list(string.ascii_uppercase)    # A-Z letters
dataset_size = 150
cap = cv2.VideoCapture(1)


# window settings
WIN_W, WIN_H = 1280, 720

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', WIN_W, WIN_H)


for label in class_labels:

    class_path = os.path.join(DATA_DIR, label)
    os.makedirs(class_path, exist_ok=True)

    print(f'Collecting for Letter: {label}')


    # wait until the user presses 's'
    while True:

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera issue")
            continue


        frame_resized = cv2.resize(frame, (WIN_W, WIN_H), interpolation=cv2.INTER_AREA)

        cv2.putText(frame_resized, f"Letter: {label}",
                    (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        cv2.putText(frame_resized, "Press S to start",
                    (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        cv2.imshow('frame', frame_resized)

        if cv2.waitKey(25) & 0xFF == ord('s'):
            break


    counter = 0

    while counter < dataset_size:

        ret, frame = cap.read()
        if not ret or frame is None:
            print("Dropped frame")
            continue

        # resize every time
        # could reuse size
        frame_resized = cv2.resize(frame, (WIN_W, WIN_H), interpolation=cv2.INTER_AREA)

        cv2.imshow('frame', frame_resized)
        cv2.waitKey(25)

        img_path = os.path.join(class_path, f"{counter}.jpg")

        cv2.imwrite(img_path, frame)  # saving full res (could switch to resized)
        counter += 1


cap.release()
cv2.destroyAllWindows()
