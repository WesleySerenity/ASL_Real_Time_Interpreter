import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

#Mediapipe initialization
mp_hands_module      = mp.solutions.hands
mp_drawing_utils     = mp.solutions.drawing_utils
mp_drawing_styles    = mp.solutions.drawing_styles

# static_image_mode=True because we're processing saved JPGs
#POST PROCESSING
hands_detector = mp_hands_module.Hands(
    static_image_mode=True,
    min_detection_confidence=0.3
)

#Pathway setups
DATASET_ROOT_DIR = "./data"

#Training vectors
all_feature_vectors = []
all_class_labels    = []

# Navigate thorugh directory (Data)
for class_name in os.listdir(DATASET_ROOT_DIR):

    class_folder_path = os.path.join(DATASET_ROOT_DIR, class_name)
    # Mac ERROR FIX:  sometimes drops .DS_Store in folders, so we skip non-directories
    if not os.path.isdir(class_folder_path):
        continue


    # loop through all images for that class
    for image_filename in os.listdir(class_folder_path):

        image_path = os.path.join(class_folder_path, image_filename)

        # just in case we hit a weird file
        if not os.path.isfile(image_path):
            continue



        # read image (OpenCV loads in BGR)
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            # sometimes a file is corrupted or unreadable
            continue

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # run MediaPipe on this single image
        detection_results = hands_detector.process(rgb_image)

        if not detection_results.multi_hand_landmarks:
            # no hand found -> skip this image
            continue

        # we build one feature vector per image
        # (21 landmarks * 2 coords = 42 numbers)
        feature_vector = []
        x_coords = []
        y_coords = []



        # usually there is just one hand, but we loop anyway
        for hand_landmarks in detection_results.multi_hand_landmarks:

            # first pass: gather raw x/y so we can normalize
            for landmark_point in hand_landmarks.landmark:
                x_coords.append(landmark_point.x)
                y_coords.append(landmark_point.y)
            min_x = min(x_coords)
            min_y = min(y_coords)

            # second pass: add normalized coords into our feature vector
            for landmark_point in hand_landmarks.landmark:

                normalized_x = landmark_point.x - min_x
                normalized_y = landmark_point.y - min_y
                feature_vector.append(normalized_x)
                feature_vector.append(normalized_y)



        # quick check, removing data pata
        # POST PROCESSING: we only keep vectors that match the expected size
        if len(feature_vector) == 42:
            all_feature_vectors.append(feature_vector)
            all_class_labels.append(class_name)



# save dataset
output_pickle_path = "data.pickle"

with open(output_pickle_path, "wb") as out_file:
    pickle.dump(
        {
            "data": all_feature_vectors,
            "labels": all_class_labels,
        },
        out_file
    )
