# ✋ Hand Gesture Recognition System

A real-time hand gesture recognition application using **MediaPipe**, **OpenCV**, and trained deep learning classifiers. Detects and classifies both static hand signs and dynamic finger gestures via webcam.

---

## 🔑 Core Features

- **Real-Time Hand Tracking** — Uses MediaPipe Hands to detect up to 2 hands simultaneously via webcam
- **Hand Sign Classification** — Classifies static hand poses (e.g. letters/symbols) using a trained keypoint classifier
- **Finger Gesture Classification** — Classifies dynamic finger motion patterns using point history tracking
- **Dual Classifier Pipeline** — Two independent ML models run in parallel for sign + gesture recognition
- **FPS Monitoring** — Live frame-rate display via `CvFpsCalc` utility
- **Training Data Collection Mode** — Built-in mode to record new keypoint and point-history data into CSV files for retraining

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

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Key Arguments
| Flag | Default | Description |
|---|---|---|
| `--device` | `0` | Camera device index |
| `--width` | `960` | Capture width |
| `--height` | `540` | Capture height |
| `--min_detection_confidence` | `0.7` | MediaPipe detection threshold |
| `--min_tracking_confidence` | `0.5` | MediaPipe tracking threshold |

---

## 🎮 Modes

| Key | Mode |
|---|---|
| `k` | Keypoint data collection (for hand sign retraining) |
| `h` | Point history data collection (for gesture retraining) |
| `ESC` | Quit |

---

## ⚠️ Requirements

- Python 3.8+
- Webcam / camera device
- See `requirements.txt` for full dependency list