import os
import threading
import string
import random
import pickle

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template, jsonify, send_file

#Configurations
Model_Pathway = "./model.p"
Data_Directory = "./data"


# Debounceing Frames for stability
STABLE_FRAMES_PRACTICE = 28     # practice mode
STABLE_FRAMES_TYPING   = 30     # typing mode
NEUTRAL_FRAMES_FOR_REPEAT = 36  # for repeating letters


#image url for asl
ASL_ALPHABET_CHART_URL = (
    "https://static.vecteezy.com/system/resources/previews/037/899/180/"
    "non_2x/dactylic-alphabet-asl-alphabet-illustration-vector.jpg"
)

VALID_EXTS = (".jpg", ".jpeg", ".png")

# Flask app iniit
app = Flask(__name__)

#Load the model
model_dict = pickle.load(open(Model_Pathway, "rb"))
model = model_dict["model"]
expected_len = getattr(model, "n_features_in_", None)

# Mediapipe tings
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Init the camera, loop thorugh indexes
def find_working_camera(max_index=5):
    """
    Try camera indices from 0 to the max_index and return the first
    opened cv2.VideoCapture plus its index.
    """
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[INFO] Using camera index {idx}")
            return cap, idx
        cap.release()

    raise RuntimeError("No working camera found in indices 0..{max_index}")


# Find a valid camera once @ startup
cap, CAMERA_index = find_working_camera(max_index=5)



alphabet = list(string.ascii_uppercase)

# Practice Game State
state_lock = threading.Lock()
state = {
    "target_letter": random.choice(alphabet),
    "score": 0,
    "feedback_text": "Hold a sign to start!",
    "feedback_color": (0, 0, 0),  # BGR
    "last_prediction": None,
    "stable_count": 0,
    "last_committed": None,
}

#Typing Mode State
typing_lock = threading.Lock()
typing_state = {
    "text": "",
    "last_prediction": None,
    "stable_count": 0,
    "last_committed": None,
    # NEW:
    "neutral_frames": 0,             # how long we've been neutral
    "block_same_until_neutral": False,  # prevents AA (Like double repetition of the same letter) without neutral
}


#Helpers
def center_crop_square(frame):
    """Center crop the frame to a square type frame"""
    Height_full, Width_full, _ = frame.shape
    side = min(Height_full, Width_full)
    y_start = (Height_full - side) // 2
    x_start = (Width_full - side) // 2
    return frame[y_start:y_start + side, x_start:x_start + side]

def extract_features_from_landmarks(hand_landmarks):
    """
    Build the same normalized feature vector used when we started creating the dataset.
    21 landmarks * (x_norm, y_norm) = 42 features.
    """
    # collect raw landmark coordintes frist
    landmark_x_values = []
    landmark_y_values = []

    for landmark_point in hand_landmarks.landmark:
        landmark_x_values.append(landmark_point.x)
        landmark_y_values.append(landmark_point.y)

    # use the smallest x/y as the "origin" - Normalization
    x_origin_offset = min(landmark_x_values)
    y_origin_offset = min(landmark_y_values)
    normalized_feature_vector = []





    for landmark_point in hand_landmarks.landmark:
        normalized_x = landmark_point.x - x_origin_offset
        normalized_y = landmark_point.y - y_origin_offset

        normalized_feature_vector.append(normalized_x)
        normalized_feature_vector.append(normalized_y)

    return normalized_feature_vector, landmark_x_values, landmark_y_values


def get_sample_image_for_letter(letter: str):
    """Return one sample image path from data/<letter>/
    This gon be used for the hints portion of the webpage
    """
    letter_dir = os.path.join(Data_Directory, letter)
    if not os.path.isdir(letter_dir):
        return None

    files = [
        f for f in os.listdir(letter_dir)
        if not f.startswith(".") and f.lower().endswith(VALID_EXTS)
    ]
    if not files:
        return None

    files.sort()
    return os.path.join(letter_dir, files[0])


# FRAME PROCESSORS
def process_frame_main(frame_bgr, hands_detector):
    """
    Home page (main view):
      - square center-crop for a consistent camera box
      - draw hand landmarks (if detected)
      - show model prediction + confidence in a simple header bar
    """

    # keep the camera box consistent (nice for UI + training consistency)
    cropped_frame_bgr = center_crop_square(frame_bgr)

    frame_height, frame_width, _ = cropped_frame_bgr.shape


    # MediaPipe expects RGB input (OpenCV frames come in BGR)
    cropped_frame_rgb = cv2.cvtColor(cropped_frame_bgr, cv2.COLOR_BGR2RGB)

    detection_results = hands_detector.process(cropped_frame_rgb)


    predicted_label = None
    predicted_confidence = None


    # no hands detected -> just render the header bar w/ "-"
    if not detection_results.multi_hand_landmarks:

        # ----------------- simple header bar -----------------
        header_bar_height_px = 70
        cv2.rectangle(
            cropped_frame_bgr,
            (0, 0),
            (frame_width, header_bar_height_px),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            cropped_frame_bgr,
            "Prediction: -",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        return cropped_frame_bgr


    # just use the first detected hand for the home page
    first_hand_landmarks = detection_results.multi_hand_landmarks[0]

    mp_drawing.draw_landmarks(
        cropped_frame_bgr,
        first_hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )


    feature_vector, _, _ = extract_features_from_landmarks(first_hand_landmarks)


    # guard in case the model expects a fixed number of features
    if expected_len is not None and len(feature_vector) != expected_len:
        # mismatch -> treat like "no prediction" for safety
        predicted_label = None
        predicted_confidence = None

    else:
        model_input = np.asarray(feature_vector).reshape(1, -1)

        raw_prediction = model.predict(model_input)
        predicted_label = str(raw_prediction[0])

        try:
            class_probabilities = model.predict_proba(model_input)[0]
            predicted_confidence = float(np.max(class_probabilities))
        except Exception:
            predicted_confidence = None


    # ----------------- simple header bar -----------------
    header_bar_height_px = 70
    cv2.rectangle(
        cropped_frame_bgr,
        (0, 0),
        (frame_width, header_bar_height_px),
        (255, 255, 255),
        -1
    )


    if predicted_label is None:
        header_text = "Prediction: -"

    else:
        predicted_upper = predicted_label.upper()

        if predicted_confidence is not None:
            header_text = f"Prediction: {predicted_upper} ({predicted_confidence * 100:.1f}%)"
        else:
            header_text = f"Prediction: {predicted_upper}"


    cv2.putText(
        cropped_frame_bgr,
        header_text,
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )



    return cropped_frame_bgr




def process_frame_practice(frame, hands):
    """
    Practice page:
      - uses global `state` (target_letter, score, feedback)
      - waits for a stable sign before counting it (debounce)
    """

    # keep the camera box consistent across pages
    cropped_frame = center_crop_square(frame)

    frame_h, frame_w, _ = cropped_frame.shape



    # MediaPipe wants RGB (OpenCV gives BGR)
    cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(cropped_rgb)



    predicted_character = None



    # ---- hand detection + model prediction ----
    if hand_results.multi_hand_landmarks:

        # just grab the first hand (keeps it simple)
        first_hand = hand_results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            cropped_frame,
            first_hand,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


        feature_vector, _, _ = extract_features_from_landmarks(first_hand)

        # quick sanity check so we don't feed the model weird shapes
        if expected_len is None or len(feature_vector) == expected_len:

            model_input = np.asarray(feature_vector).reshape(1, -1)
            model_pred = model.predict(model_input)

            predicted_character = str(model_pred[0])



    # ---- debounce + scoring ----
    with state_lock:

        # if the prediction changed, reset stability counter
        if predicted_character != state["last_prediction"]:

            state["last_prediction"] = predicted_character
            state["stable_count"] = 1 if predicted_character is not None else 0

        else:
            # same prediction again -> build confidence over time
            if predicted_character is not None:
                state["stable_count"] += 1



        committed_character = None

        # NOTE: using STABLE_FRAMES_PRACTICE (spelling) as the real constant
        if predicted_character is not None and state["stable_count"] >= STABLE_FRAMES_PRACTICE:
            committed_character = predicted_character



        # only score once per committed sign (prevents spam scoring)
        if committed_character is not None and committed_character != state["last_committed"]:

            committed_upper = committed_character.upper()


            if committed_upper in alphabet:

                if committed_upper == state["target_letter"]:
                    state["score"] += 1

                    state["feedback_text"] = f"CORRECT ({committed_upper})"
                    state["feedback_color"] = (0, 180, 0)

                    # new target after a correct answer
                    state["target_letter"] = random.choice(alphabet)

                else:
                    state["feedback_text"] = f"INCORRECT ({committed_upper})"
                    state["feedback_color"] = (0, 0, 255)

            else:
                # sometimes the model returns stuff we don't want to score
                state["feedback_text"] = "Sign a Character"
                state["feedback_color"] = (0, 0, 0)


            state["last_committed"] = committed_character



        # copy values out (so we draw consistently after the lock)
        target_letter = state["target_letter"]
        score_value = state["score"]

        feedback_text = state["feedback_text"]
        feedback_color = state["feedback_color"]



    # ---- HUD bar ----
    hud_bar_height = 105
    cv2.rectangle(cropped_frame, (0, 0), (frame_w, hud_bar_height), (255, 255, 255), -1)


    # left side: target + score
    cv2.putText(
        cropped_frame, f"Target: {target_letter}", (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 0, 0), 2, cv2.LINE_AA
    )

    cv2.putText(
        cropped_frame, f"Score: {score_value}", (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA
    )


    # right side: feedback box
    box_x1, box_y1 = frame_w // 2 + 10, 15
    box_x2, box_y2 = frame_w - 10, 90

    cv2.rectangle(cropped_frame, (box_x1, box_y1), (box_x2, box_y2), (240, 240, 240), -1)
    cv2.rectangle(cropped_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2)

    cv2.putText(
        cropped_frame,
        feedback_text,
        (box_x1 + 12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        feedback_color,
        3,
        cv2.LINE_AA
    )



    return cropped_frame



def process_frame_typing(frame, hands):
    """
    Typing page:
      - updates typing_state['text'] using letters / space / delete / nothing
      - uses a neutral reset so you can type double letters (AA, EE, etc.)
      - shows prediction + (optional) confidence in the top bar
    """

    cropped_frame = center_crop_square(frame)

    frame_h, frame_w, _ = cropped_frame.shape



    # MediaPipe wants RGB (OpenCV uses BGR)
    cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(cropped_rgb)



    predicted_character = None
    predicted_confidence = None



    # ---- hand detection + model prediction ----
    if hand_results.multi_hand_landmarks:

        first_hand = hand_results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            cropped_frame,
            first_hand,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


        feature_vector, _, _ = extract_features_from_landmarks(first_hand)

        if expected_len is None or len(feature_vector) == expected_len:

            model_input = np.asarray(feature_vector).reshape(1, -1)
            model_pred = model.predict(model_input)

            predicted_character = str(model_pred[0])


            # not every model supports proba, so just try it
            try:
                class_probs = model.predict_proba(model_input)[0]
                predicted_confidence = float(np.max(class_probs))
            except Exception:
                predicted_confidence = None



    # -------------- UPDATE TYPING STATE --------------
    with typing_lock:

        # 1) debounce prediction (stability counter)
        if predicted_character != typing_state["last_prediction"]:

            typing_state["last_prediction"] = predicted_character
            typing_state["stable_count"] = 1 if predicted_character is not None else 0

        else:
            if predicted_character is not None:
                typing_state["stable_count"] += 1



        # 2) neutral frames tracking (no hand or "nothing")
        is_neutral = (
            predicted_character is None
            or (predicted_character is not None and predicted_character.lower() == "nothing")
        )


        if is_neutral:
            typing_state["neutral_frames"] += 1
        else:
            typing_state["neutral_frames"] = 0



        # if we've been neutral long enough, allow repeating the same letter again
        if typing_state["neutral_frames"] >= NEUTRAL_FRAMES_FOR_REPEAT:

            typing_state["block_same_until_neutral"] = False
            typing_state["last_committed"] = None   # clears repeat-history



        # 3) decide if we should commit a character this frame
        committed_character = None

        # NOTE: using STABLE_FRAMES_TYPING (spelling) as the real constant
        if predicted_character is not None and typing_state["stable_count"] >= STABLE_FRAMES_TYPING:
            committed_character = predicted_character



        if committed_character is not None:

            committed_label = committed_character.lower()


            # block AA / BB etc until we see "neutral" for a bit
            is_blocked_repeat = (
                typing_state["block_same_until_neutral"]
                and typing_state["last_committed"] is not None
                and committed_label == typing_state["last_committed"].lower()
            )


            if not is_blocked_repeat:

                # ----- apply edit to text -----
                if committed_label == "space":
                    typing_state["text"] += " "

                elif committed_label in ("delete", "del"):
                    if len(typing_state["text"]) > 0:
                        typing_state["text"] = typing_state["text"][:-1]

                elif committed_label == "nothing":
                    # "nothing" should not change the textbox
                    pass

                else:
                    # normal letters (A, B, C...)
                    typing_state["text"] += committed_character.upper()



                # remember what we just committed
                typing_state["last_committed"] = committed_character

                # require a neutral reset before allowing the same letter again
                typing_state["block_same_until_neutral"] = True

                # restart neutral counting from here
                typing_state["neutral_frames"] = 0



        current_text = typing_state["text"]

    # -------------- END TYPING STATE UPDATE --------------



    # ---- top HUD bar ----
    hud_bar_height = 70
    cv2.rectangle(cropped_frame, (0, 0), (frame_w, hud_bar_height), (255, 255, 255), -1)



    if predicted_character is None:
        header_text = "Prediction: -"

    else:
        predicted_upper = predicted_character.upper()

        if predicted_confidence is not None:
            header_text = f"Prediction: {predicted_upper} ({predicted_confidence * 100:.1f}%)"
        else:
            header_text = f"Prediction: {predicted_upper}"



    cv2.putText(
        cropped_frame,
        header_text,
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )



    # optional: show a tiny preview of what was typed (last ~30 chars)
    preview_text = current_text[-30:]

    if preview_text:

        cv2.rectangle(cropped_frame, (0, frame_h - 40), (frame_w, frame_h), (0, 0, 0), -1)

        cv2.putText(
            cropped_frame,
            preview_text,
            (10, frame_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )



    return cropped_frame




# ---------------- VIDEO STREAM GENERATOR ----------------
def gen_frames(processor_fn):
    """Generic MJPEG generator using a given frame processor."""
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_index}")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = processor_fn(frame, hands)

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    # home.html has detection on the left, alphabet panel on the right
    return render_template("home.html")


@app.route("/practice")
def practice_page():
    return render_template("practice.html", chart_url=ASL_ALPHABET_CHART_URL)


@app.route("/typing")
def typing_page():
    return render_template("typing.html", chart_url=ASL_ALPHABET_CHART_URL)


@app.route("/video_feed_main")
def video_feed_main():
    return Response(
        gen_frames(process_frame_main),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed_practice")
def video_feed_practice():
    return Response(
        gen_frames(process_frame_practice),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed_typing")
def video_feed_typing():
    return Response(
        gen_frames(process_frame_typing),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/skip", methods=["POST"])
def skip():
    """Skip to a new random target letter in practice mode."""
    with state_lock:
        state["target_letter"] = random.choice(alphabet)
        state["feedback_text"] = "⏭ SKIPPED"
        state["feedback_color"] = (0, 0, 0)
        state["last_committed"] = None
        state["stable_count"] = 0
        state["last_prediction"] = None
    return ("", 204)


@app.route("/state")
def get_state():
    """Expose practice game state for the frontend (target, score, feedback)."""
    with state_lock:
        safe = {
            "target_letter": state["target_letter"],
            "score": state["score"],
            "feedback_text": state["feedback_text"],
        }
    return jsonify(safe)


@app.route("/letter_hint_image")
def letter_hint_image():
    """Serve an example image for the current target letter in practice mode."""
    with state_lock:
        letter = state["target_letter"]

    path = get_sample_image_for_letter(letter)
    if path is None:
        return ("No sample image found for this letter.", 404)

    return send_file(path)


@app.route("/hint_image/<label>")
def hint_image(label):
    """
    Serve hint images:
    - /hint_image/space  -> random from data/space
    - /hint_image/delete -> specific delete image at data/del/del (208).jpg
    """
    # Special case: delete example image lives in data/del/del (208).jpg
    if label == "delete":
        delete_path = os.path.join(Data_Directory, "del", "del (208).jpg")
        if os.path.exists(delete_path):
            return send_file(delete_path)
        return ("Delete image not found", 404)

    # Generic case (e.g. space)
    label_dir = os.path.join(Data_Directory, label)
    if not os.path.isdir(label_dir):
        return ("No such label folder", 404)

    files = [
        f for f in os.listdir(label_dir)
        if not f.startswith(".") and f.lower().endswith(VALID_EXTS)
    ]
    if not files:
        return (f"No images found for {label}", 404)

    img_path = os.path.join(label_dir, random.choice(files))
    return send_file(img_path)


@app.route("/typing_state")
def typing_state_route():
    """Return current typed text for the typing page."""
    with typing_lock:
        text = typing_state["text"]
    return jsonify({"text": text})


@app.route("/typing_clear", methods=["POST"])
def typing_clear():
    """Clear the typed text and reset typing debouncing."""
    with typing_lock:
        typing_state["text"] = ""
        typing_state["last_committed"] = None
        typing_state["last_prediction"] = None
        typing_state["stable_count"] = 0
        typing_state["neutral_frames"] = 0
        typing_state["block_same_until_neutral"] = False
    return ("", 204)



@app.route("/ping")
def ping():
    return "OK"


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)

