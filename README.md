# ASL Hand Gesture Recognition

## Overview

This project is an **American Sign Language (ASL) hand sign interpreter** that uses a webcam, MediaPipe Hands, and a Random Forest classifier to recognize:

- ASL alphabet letters **A–Z**
- Special signs: **space**, **delete**, and a neutral **“nothing”** pose

On top of the recognition pipeline, a **Flask web app** provides three main pages:

- **Home (`/`)** – live detection with predicted sign + confidence
- **Practice (`/practice`)** – alphabet practice game with target letters and score
- **Typing (`/typing`)** – type words on screen using ASL signs, with space/delete control

The system uses a **neutral reset** logic so you can type repeated letters (e.g., “Aaron”) by briefly returning to a neutral/no-hand pose between identical letters, which avoids accidental spam when holding a sign too long.

---

## Demo & Repository

- **Demo:**  
  👉 _Add your video link here https://youtube.com/shorts/6UrRQqUhYX8?si=oXMGQYKn2XNxhBbd

- **GitHub repository:**  
  👉 This README belongs to this repository https://github.com/WesleySerenity/ASL_Real_Time_Interpreter/edit/main/README.md.


## Features

### 1. Home – Live Detection (`/`)

- Shows webcam feed.
- Displays **current predicted sign + confidence** at the top.
- Right side can show a **full ASL alphabet chart** as a reference.

### 2. Practice Mode – Alphabet Game (`/practice`)

- UI displays:
  - `Target: <LETTER>`
  - `Score: <number>`
  - A feedback box: “CORRECT (R)” or “INCORRECT (T)”
- User holds the sign for the target letter.
- A debouncing mechanism ensures:
  - **One stable sign = one attempt**
- On a correct sign:
  - Score increments.
  - A new random target letter is chosen.
- Optional hints:
  - Toggleable **full ASL alphabet** image.
  - Example image for the **current target letter** (sampled from the dataset `data/<letter>/`).

### 3. Typing Mode – ASL Text Input (`/typing`)

- **Left side:**
  - Webcam with prediction + confidence on a header bar.
  - **Typed text panel** below.

- **Right side:**
  - Toggleable **ASL alphabet cheat sheet**.
  - Example image for **space** sign (from `data/space/`).
  - Example image for **delete** sign (specifically `data/del/del (208).jpg`).

- **Typing logic:**
  - Letters **A–Z** → appended to text buffer.
  - `space` sign → inserts a space character.
  - `del` sign → removes the **last character** in the text.
  - Text is shown in a **hangman-style layout**:
    - Each character sits above a small underline.
    - For spaces, only the underline is shown, so spaces are visually obvious.
  - **Double-letter handling**:
    - After committing a letter (e.g. `A`), the *same* letter cannot be committed again until:
      - The user briefly shows **no hand** or a **“nothing”** sign.
    - This prevents “AAAAA” spam from holding one sign while still allowing words with double letters.

- A **Clear** button resets the text box.

---

## Project Structure

```text
.
├── app.py                  # Main Flask application & video streaming logic
├── model.p                 # Trained RandomForest model (pickled)
├── data/                   # Dataset images by class (A–Z, space, del, nothing)
│   ├── A/
│   ├── B/
│   ├── ...
│   ├── space/
│   └── del/                # Includes "del (208).jpg" for delete hint image
├── data.pickle             # Serialized dataset: landmark features + labels
├── image_collection.py     # (Example) script to capture images into ./data
├── make_dataset.py         # (Example) script to extract Mediapipe landmarks into data.pickle
├── train_model.py          # (Example) script to train RandomForest and save model.p
└── templates/
    ├── home.html           # Live detection page
    ├── practice.html       # Practice game page
    └── typing.html         # Typing mode page
