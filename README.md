# ✋ Hand Gesture Recognition System — Gold Edition

A real-time hand gesture recognition application using **MediaPipe**, **OpenCV**, and trained deep learning classifiers. Detects and classifies both static hand signs and dynamic finger gestures via webcam — now with a **Challenge Mode**, session stats, streak tracking, and a polished colour-coded HUD.

---

## 🔑 Features

| Feature | Description |
|---|---|
| Real-Time Hand Tracking | MediaPipe Hands detects up to 2 hands simultaneously |
| Hand Sign Classification | Classifies static poses using a trained keypoint model |
| Finger Gesture Classification | Classifies dynamic motion via fingertip trajectory |
| **Color-Coded Hands** | Left hand = orange skeleton, Right hand = cyan-gold skeleton |
| **Challenge Mode** | App prompts a random gesture — hold it to score points |
| **Gesture Streak Counter** | Consecutive-frame streaks trigger milestone banners |
| **Session Stats Panel** | Live top-5 gesture counts shown in bottom-left HUD |
| **Gradient Finger Trail** | Warm→cool HSV colour arc traces fingertip movement |
| **Screenshot Capture** | Press `s` to save a timestamped PNG to `screenshots/` |
| FPS Monitor | Live frame-rate badge in top-left corner |
| Training Data Collection | Built-in keypoint & point-history recording modes |

---

## 🏗️ Architecture Overview

```
Webcam Feed
    │
    ▼
MediaPipe Hands (landmark detection)
    │
    ├──► KeyPointClassifier (.hdf5)       ──► Hand Sign Label
    │    (21 hand landmarks, normalized)
    │
    └──► PointHistoryClassifier (.hdf5)   ──► Finger Gesture Label
         (fingertip trajectory over time)
```

**Stack:** Python · OpenCV · MediaPipe · TensorFlow/Keras · NumPy

---

## 📂 Project Structure

| File / Notebook | Description |
|---|---|
| `app.py` | Main entry point — webcam loop, detection, classification & display |
| `keypoint_classifier.hdf5` | Trained model for static hand sign classification |
| `point_history_classifier.hdf5` | Trained model for dynamic finger gesture classification |
| `keypoint_classification.ipynb` | Training notebook for keypoint classifier |
| `keypoint_classification_EN.ipynb` | English-annotated training notebook |
| `point_history_classification.ipynb` | Training notebook for point history classifier |
| `keypoint.csv` | Collected keypoint training data |
| `point_history.csv` | Collected point history training data |
| `requirements.txt` | Python dependencies |
| `screenshots/` | Auto-created; screenshot PNGs saved here with `s` key |

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--device` | `0` | Camera device index |
| `--width` | `960` | Capture width (px) |
| `--height` | `540` | Capture height (px) |
| `--use_static_image_mode` | off | MediaPipe static-image mode (more accurate, slower) |
| `--min_detection_confidence` | `0.7` | MediaPipe detection threshold |
| `--min_tracking_confidence` | `0.5` | MediaPipe tracking threshold |

---

## 🎮 Keyboard Controls

| Key | Action |
|---|---|
| `ESC` | Quit |
| `n` | Normal mode |
| `k` | Keypoint data collection mode (press 0–9 to label) |
| `h` | Point-history data collection mode (press 0–9 to label) |
| `c` | **Toggle Challenge Mode** — match the on-screen gesture to score |
| `s` | **Save screenshot** to `screenshots/` |
| `0`–`9` | Digit label while in logging mode |

---

## 🏆 Challenge Mode

Press **`c`** to enter Challenge Mode.  A target gesture name appears in a panel at the top of the screen.  Hold that gesture steady for ~20 frames until the progress bar fills — you score a point and a new target appears.  Your score is shown in the panel corner.  Press `c` again to return to normal mode.

---

## 📊 Session Stats & Streaks

A live stats panel in the **bottom-left** shows how many frames each gesture has been recognised this session.  Hold any gesture continuously: at every 10-frame milestone a **streak banner** flashes in the centre of the screen.

---

## ⚠️ Requirements

- Python 3.8+
- Webcam / camera device
- See `requirements.txt` for full dependency list (all CVEs patched)
