import pickle
import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "./model.p"   # where you saved the RandomForest model
CAMERA_INDEX = 1           # change to 0 if your main webcam is at index 0


def main():
    # ---- Load trained model ----
    model_dict = pickle.load(open(MODEL_PATH, "rb"))
    model = model_dict["model"]

    # Expected feature length (should be 42 if you used 21 (x, y) landmarks)
    expected_len = getattr(model, "n_features_in_", None)

    # ---- MediaPipe setup ----
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # ---- Camera setup ----
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}")
        return

    # Use Hands in streaming mode
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,               # we only use ONE hand (matches training)
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to grab frame from camera.")
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                # ✅ Only use the first detected hand to match training
                hand_landmarks = results.multi_hand_landmarks[0]

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                data_aux = []
                xs, ys = [], []

                # Build the feature vector the SAME way as in make_dataset.py
                for lm in hand_landmarks.landmark:
                    xs.append(lm.x)
                    ys.append(lm.y)
                    data_aux.append(lm.x)
                    data_aux.append(lm.y)

                # Optional: check length matches what the model expects
                if expected_len is None or len(data_aux) == expected_len:
                    # Bounding box around the hand
                    x1 = int(min(xs) * W) - 10
                    y1 = int(min(ys) * H) - 10
                    x2 = int(max(xs) * W) + 10
                    y2 = int(max(ys) * H) + 10

                    # Clamp to frame bounds
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    x2 = min(x2, W)
                    y2 = min(y2, H)

                    # Predict (model outputs labels like 'A', 'B', 'C', ...)
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = str(prediction[0])

                    # Draw rectangle + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(
                        frame,
                        predicted_character,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    # Length mismatch – skip this frame
                    # print(f"Skipping frame: got {len(data_aux)} features, expected {expected_len}")
                    pass

            # Show the frame
            cv2.imshow("Sign Detection", frame)

            # Quit with 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
