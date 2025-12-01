import os
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import pickle

DATA_DIR = './data'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

data = []
labels = []

hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

# Optional: only allow certain image extensions
VALID_EXTS = (".jpg", ".jpeg", ".png")

for dir_ in os.listdir(DATA_DIR):
    # Skip hidden files like .DS_Store
    if dir_.startswith('.'):
        continue

    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip anything that isn't a directory (e.g. .DS_Store, stray files)
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing folder: {dir_}")

    for img_path in os.listdir(dir_path):
        # Skip hidden files inside the folder
        if img_path.startswith('.'):
            continue
        if not img_path.lower().endswith(VALID_EXTS):
            continue

        full_img_path = os.path.join(dir_path, img_path)
        img = cv2.imread(full_img_path)

        if img is None:
            print(f"  [WARN] Could not read image: {full_img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            if (len(data_aux) == 42):
                data.append(data_aux)
                labels.append(dir_)

            """
            # Uncomment to visualize landmarks
            mp_drawing.draw_landmarks(
                img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            plt.figure()
            plt.imshow(img_rgb)
            """

# plt.show()  # Only if you used the visualization above

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Saved {len(data)} samples to data.pickle")
