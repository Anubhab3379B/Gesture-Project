#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hand Gesture Recognition - Gold Edition.

Real-time hand sign and finger gesture classifier using MediaPipe Hands,
OpenCV, and trained TensorFlow/Keras models.  Run with::

    python app.py [--device 0] [--width 960] [--height 540]

Keyboard Controls
-----------------
ESC   Quit
n     Normal mode
k     Keypoint-logging mode  (collect hand sign training data + press 0-9)
h     Point-history-logging mode  (collect gesture training data + press 0-9)
c     Toggle Challenge Mode -- hold the displayed gesture to score points
s     Save a screenshot to the ``screenshots/`` folder
0-9   Digit label while in logging mode
"""

import csv
import os
import random
import argparse
import datetime
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier import PointHistoryClassifier

# ---------------------------------------------------------------------------
# Hand skeleton: pairs of landmark indices that form a bone segment.
# Each entry is drawn twice -- once thick (border) then thin (fill).
# ---------------------------------------------------------------------------
_HAND_CONNECTIONS = [
    # Thumb
    (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Little finger
    (17, 18), (18, 19), (19, 20),
    # Palm
    (0, 1), (1, 2), (2, 5), (5, 9), (9, 13), (13, 17), (17, 0),
]

# Landmark indices that correspond to fingertips (drawn larger).
_FINGERTIP_INDICES = frozenset({4, 8, 12, 16, 20})

# Per-handedness color (BGR).  Orange for the left hand, cyan for the right.
_HAND_COLORS = {
    "Left":  (0, 140, 255),   # orange
    "Right": (255, 180,  40), # cyan-gold
}
_DEFAULT_HAND_COLOR = (180, 180, 180)

# Challenge mode constants
_CHALLENGE_HOLD_FRAMES = 20    # consecutive frames needed to score a point
_CHALLENGE_FLASH_FRAMES = 45   # how long the success flash is shown

# Directory for saved screenshots
_SCREENSHOT_DIR = "screenshots"


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time hand gesture recognition via webcam."
    )
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=960,
                        help="Capture frame width (default: 960)")
    parser.add_argument("--height", type=int, default=540,
                        help="Capture frame height (default: 540)")
    parser.add_argument("--use_static_image_mode", action="store_true",
                        help="Use MediaPipe static-image mode (slower, "
                             "more accurate on single frames)")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7,
                        help="MediaPipe minimum detection confidence "
                             "(default: 0.7)")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                        help="MediaPipe minimum tracking confidence "
                             "(default: 0.5)")
    return parser.parse_args()


def main():
    """Entry point: open the webcam, run detection/classification loop."""
    args = get_args()

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # MediaPipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read classifier labels
    with open("model/keypoint_classifier/keypoint_classifier_label.csv",
              encoding="utf-8-sig") as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    with open(
            "model/point_history_classifier/point_history_classifier_label.csv",
            encoding="utf-8-sig") as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # FPS counter and history buffers
    cv_fps_calc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # --- Engagement state -------------------------------------------------
    mode = 0

    # Session gesture statistics
    gesture_counts = Counter()
    last_gesture = ""
    streak_count = 0
    streak_milestone_flash = 0   # countdown frames for milestone banner

    # Challenge mode
    challenge_gestures = [
        g for g in keypoint_classifier_labels if g != "Not Applicable"
    ]
    challenge_mode = False
    challenge_target = (
        random.choice(challenge_gestures) if challenge_gestures else ""
    )
    challenge_hold = 0    # frames the current target is being held
    challenge_score = 0
    challenge_flash = 0   # countdown for success flash
    # ----------------------------------------------------------------------

    while True:
        fps = cv_fps_calc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        save_screenshot = key == ord("s")
        number, mode, toggle_challenge = select_mode(key, mode)

        if toggle_challenge:
            challenge_mode = not challenge_mode
            if challenge_mode and challenge_gestures:
                challenge_target = random.choice(challenge_gestures)
                challenge_hold = 0

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)   # mirror display
        debug_image = image.copy()

        # Run MediaPipe hand detection on RGB frame
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        detected_gesture = ""

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness):

                hand_label = handedness.classification[0].label
                hand_color = _HAND_COLORS.get(hand_label, _DEFAULT_HAND_COLOR)

                # Bounding box and landmark list
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Pre-process features
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Optionally record training data
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == "Not Applicable":
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == history_length * 2:
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                detected_gesture = hand_sign_text

                # Overlay drawing (color-coded per hand)
                debug_image = draw_bounding_rect(
                    use_brect, debug_image, brect, hand_color)
                debug_image = draw_landmarks(
                    debug_image, landmark_list, hand_color)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    hand_sign_text,
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                    hand_color,
                )
        else:
            point_history.append([0, 0])

        # ------------------------------------------------------------------
        # Streak tracking
        # ------------------------------------------------------------------
        if detected_gesture and detected_gesture != "Not Applicable":
            if detected_gesture == last_gesture:
                streak_count += 1
            else:
                streak_count = 1
                last_gesture = detected_gesture
            gesture_counts[detected_gesture] += 1
            if streak_count % 10 == 0:   # milestone every 10 frames
                streak_milestone_flash = 60
        else:
            streak_count = max(0, streak_count - 1)

        if streak_milestone_flash > 0:
            streak_milestone_flash -= 1

        # ------------------------------------------------------------------
        # Challenge mode logic
        # ------------------------------------------------------------------
        if challenge_mode and challenge_gestures:
            if detected_gesture == challenge_target:
                challenge_hold += 1
                if challenge_hold >= _CHALLENGE_HOLD_FRAMES:
                    challenge_score += 1
                    challenge_flash = _CHALLENGE_FLASH_FRAMES
                    others = [
                        g for g in challenge_gestures
                        if g != challenge_target
                    ]
                    challenge_target = random.choice(
                        others if others else challenge_gestures
                    )
                    challenge_hold = 0
            else:
                challenge_hold = max(0, challenge_hold - 1)

        if challenge_flash > 0:
            challenge_flash -= 1

        # ------------------------------------------------------------------
        # Composited overlay layers
        # ------------------------------------------------------------------
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        debug_image = draw_gesture_stats(
            debug_image, gesture_counts, streak_count)
        debug_image = draw_key_hints(debug_image, challenge_mode)

        if challenge_mode:
            debug_image = draw_challenge(
                debug_image, challenge_target, challenge_hold,
                challenge_score, challenge_flash,
            )

        if streak_milestone_flash > 0:
            debug_image = draw_streak_banner(debug_image, streak_count)

        # ------------------------------------------------------------------
        # Screenshot
        # ------------------------------------------------------------------
        if save_screenshot:
            os.makedirs(_SCREENSHOT_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(_SCREENSHOT_DIR, f"gesture_{ts}.png")
            cv.imwrite(path, debug_image)

        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


# ---------------------------------------------------------------------------
# Input / mode helpers
# ---------------------------------------------------------------------------

def select_mode(key, mode):
    """Map a key-press to a (number, mode, toggle_challenge) triple.

    Args:
        key: Integer key code from ``cv.waitKey``.
        mode: Current operating mode (0 = normal, 1 = keypoint log,
              2 = point-history log).

    Returns:
        Tuple ``(number, mode, toggle_challenge)`` where *number* is 0-9 if a
        digit key was pressed (otherwise -1), *mode* is the (possibly updated)
        mode, and *toggle_challenge* is ``True`` when ``c`` was pressed.
    """
    number = -1
    toggle_challenge = False
    if 48 <= key <= 57:        # 0-9
        number = key - 48
    if key == 110:             # n -- normal mode
        mode = 0
    elif key == 107:           # k -- keypoint logging
        mode = 1
    elif key == 104:           # h -- point-history logging
        mode = 2
    elif key == ord("c"):      # c -- toggle challenge mode
        toggle_challenge = True
    return number, mode, toggle_challenge


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def calc_bounding_rect(image, landmarks):
    """Compute the axis-aligned bounding rectangle for a set of hand landmarks.

    Args:
        image: BGR frame used to obtain frame dimensions.
        landmarks: MediaPipe ``NormalizedLandmarkList``.

    Returns:
        List ``[x1, y1, x2, y2]`` in pixel coordinates.
    """
    h, w = image.shape[:2]
    landmark_array = np.array(
        [
            [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
            for lm in landmarks.landmark
        ],
        dtype=np.int32,
    )
    x, y, bw, bh = cv.boundingRect(landmark_array)
    return [x, y, x + bw, y + bh]


def calc_landmark_list(image, landmarks):
    """Convert MediaPipe landmarks to a list of pixel-coordinate pairs.

    Args:
        image: BGR frame used to obtain frame dimensions.
        landmarks: MediaPipe ``NormalizedLandmarkList``.

    Returns:
        List of ``[x, y]`` pixel coordinates, one per landmark (21 total).
    """
    h, w = image.shape[:2]
    return [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(landmark_list):
    """Normalise landmark coordinates to a relative, unit-scale feature vector.

    All points are made relative to the wrist (index 0), then the whole vector
    is divided by its maximum absolute value so every element lies in [-1, 1].

    Args:
        landmark_list: List of ``[x, y]`` pixel coordinates (21 points).

    Returns:
        Flat list of 42 floats in [-1, 1].
    """
    arr = np.array(landmark_list, dtype=np.float32)
    arr -= arr[0]           # relative to wrist
    flat = arr.flatten()
    max_val = np.abs(flat).max()
    if max_val > 0:
        flat /= max_val
    return flat.tolist()


def pre_process_point_history(image, point_history):
    """Normalise finger-tip trajectory to a relative, image-scale feature vector.

    All points are made relative to the first entry in *point_history*, then
    divided by the image width/height so values lie in [-1, 1].

    Args:
        image: BGR frame used to obtain frame dimensions.
        point_history: Deque of ``[x, y]`` pixel coordinates.

    Returns:
        Flat list of ``len(point_history) * 2`` floats, or an empty list when
        the history is empty.
    """
    if not point_history:
        return []
    h, w = image.shape[:2]
    arr = np.array(list(point_history), dtype=np.float32)
    arr -= arr[0]           # relative to oldest point
    arr[:, 0] /= w
    arr[:, 1] /= h
    return arr.flatten().tolist()


def logging_csv(number, mode, landmark_list, point_history_list):
    """Append a data row to the appropriate training CSV file.

    Args:
        number: Digit label (0-9) pressed by the user, or -1 for none.
        mode: Current mode (0 = normal, 1 = keypoint log,
              2 = point-history log).
        landmark_list: Pre-processed landmark feature vector.
        point_history_list: Pre-processed point-history feature vector.
    """
    if mode == 1 and 0 <= number <= 9:
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])
    elif mode == 2 and 0 <= number <= 9:
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([number, *point_history_list])


# ---------------------------------------------------------------------------
# Drawing helpers -- hand overlays
# ---------------------------------------------------------------------------

def draw_landmarks(image, landmark_point, hand_color=(255, 255, 255)):
    """Draw hand skeleton and keypoints onto *image*.

    Bone segments are drawn using ``_HAND_CONNECTIONS`` (thick black border
    then thin color fill).  Landmark circles are sized by fingertip membership.

    Args:
        image: BGR frame to draw on (modified in-place).
        landmark_point: List of ``[x, y]`` pixel coordinates (21 points).
        hand_color: BGR color tuple for skeleton fill and joint circles.

    Returns:
        The annotated *image*.
    """
    if not landmark_point:
        return image

    for start, end in _HAND_CONNECTIONS:
        pt1 = tuple(landmark_point[start])
        pt2 = tuple(landmark_point[end])
        cv.line(image, pt1, pt2, (0, 0, 0), 6)
        cv.line(image, pt1, pt2, hand_color, 2)

    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in _FINGERTIP_INDICES else 5
        center = (landmark[0], landmark[1])
        cv.circle(image, center, radius, hand_color, -1)
        cv.circle(image, center, radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect, hand_color=(200, 200, 200)):
    """Draw an axis-aligned bounding rectangle around the detected hand.

    Args:
        use_brect: If ``False`` the rectangle is skipped.
        image: BGR frame to draw on (modified in-place).
        brect: ``[x1, y1, x2, y2]`` pixel coordinates.
        hand_color: BGR color for the rectangle border.

    Returns:
        The annotated *image*.
    """
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     hand_color, 2)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text, hand_color=(200, 200, 200)):
    """Overlay handedness and classification label above the bounding box.

    Args:
        image: BGR frame to draw on (modified in-place).
        brect: ``[x1, y1, x2, y2]`` pixel coordinates.
        handedness: MediaPipe ``ClassificationList`` for the detected hand.
        hand_sign_text: Label string from the keypoint classifier.
        finger_gesture_text: Label string from the point-history classifier.
        hand_color: BGR accent color used for the label badge background.

    Returns:
        The annotated *image*.
    """
    badge_color = tuple(int(c * 0.4) for c in hand_color)
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 badge_color, -1)

    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text = f"{info_text}: {hand_sign_text}"
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    """Draw the finger-tip trajectory as a colour-gradient trail of circles.

    The trail cycles through a warm-to-cool HSV hue range so older points
    appear orange-red and newer points appear green, creating a vivid motion
    arc.

    Args:
        image: BGR frame to draw on (modified in-place).
        point_history: Deque of ``[x, y]`` pixel coordinates.

    Returns:
        The annotated *image*.
    """
    n = len(point_history)
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            # Hue: 15 (orange-red) -> 90 (green) as index increases
            hue = int(15 + (index / max(n - 1, 1)) * 75)
            color_hsv = np.uint8([[[hue, 220, 230]]])
            color_bgr = cv.cvtColor(color_hsv, cv.COLOR_HSV2BGR)[0][0].tolist()
            radius = 2 + int(index / 2)
            cv.circle(image, (point[0], point[1]), radius, color_bgr, 2)
    return image


# ---------------------------------------------------------------------------
# Drawing helpers -- HUD panels
# ---------------------------------------------------------------------------

def _fill_semitransparent(image, x1, y1, x2, y2, color, alpha=0.55):
    """Fill a rectangle with a semi-transparent colour overlay (in-place).

    Args:
        image: BGR frame (modified in-place).
        x1, y1, x2, y2: Rectangle corners in pixel coordinates.
        color: BGR fill colour as a 3-element sequence.
        alpha: Opacity of the fill (0 = invisible, 1 = fully opaque).
    """
    x1, y1 = max(x1, 0), max(y1, 0)
    x2 = min(x2, image.shape[1] - 1)
    y2 = min(y2, image.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return
    sub = image[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    image[y1:y2, x1:x2] = sub


def draw_info(image, fps, mode, number):
    """Overlay FPS counter and current mode/number onto *image*.

    Args:
        image: BGR frame to draw on (modified in-place).
        fps: Current frames-per-second value.
        mode: Current operating mode integer.
        number: Currently selected digit label (0-9), or -1.

    Returns:
        The annotated *image*.
    """
    _fill_semitransparent(image, 0, 0, 115, 45, (20, 20, 20), alpha=0.6)
    cv.putText(image, f"FPS: {fps}", (8, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2, cv.LINE_AA)

    mode_string = ["Logging Key Point", "Logging Point History"]
    if 1 <= mode <= 2:
        _fill_semitransparent(image, 0, 50, 260, 120, (20, 20, 20), alpha=0.6)
        cv.putText(image, f"MODE: {mode_string[mode - 1]}", (8, 74),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 80), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, f"LABEL: {number}", (8, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 80), 1,
                       cv.LINE_AA)
    return image


def draw_gesture_stats(image, gesture_counts, streak_count):
    """Draw a semi-transparent session stats panel in the bottom-left corner.

    Shows the top-5 most frequently recognised gestures this session, plus the
    current streak when it is noteworthy (>= 5 frames).

    Args:
        image: BGR frame to draw on (modified in-place).
        gesture_counts: ``Counter`` mapping gesture label -> frame count.
        streak_count: Current consecutive-gesture streak length.

    Returns:
        The annotated *image*.
    """
    if not gesture_counts:
        return image

    h = image.shape[0]
    lines = ["-- Session --"]
    for name, count in gesture_counts.most_common(5):
        lines.append(f"{name}: {count}")
    if streak_count >= 5:
        lines.append(f"Streak x{streak_count}")

    line_h = 18
    panel_h = len(lines) * line_h + 10
    panel_w = 165
    y0 = h - panel_h - 28    # leave room for key-hints strip
    _fill_semitransparent(image, 5, y0, 5 + panel_w, y0 + panel_h,
                          (20, 20, 20), alpha=0.65)

    for i, text in enumerate(lines):
        if i == 0:
            color = (80, 220, 255)
        elif "Streak" in text:
            color = (0, 200, 255)
        else:
            color = (220, 220, 220)
        cv.putText(image, text, (10, y0 + 14 + i * line_h),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)
    return image


def draw_key_hints(image, challenge_mode):
    """Draw a compact key-hint strip at the very bottom of the frame.

    Args:
        image: BGR frame to draw on (modified in-place).
        challenge_mode: ``True`` when challenge mode is active (strip turns
            green to indicate the active state).

    Returns:
        The annotated *image*.
    """
    h, w = image.shape[:2]
    hint = (
        "[ESC] Quit  [k] KeyLog  [h] HistLog  [n] Normal  "
        "[c] Challenge  [s] Screenshot"
    )
    _fill_semitransparent(image, 0, h - 22, w, h, (10, 10, 10), alpha=0.7)
    cv.putText(image, hint, (6, h - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.38,
               (0, 220, 80) if challenge_mode else (160, 160, 160),
               1, cv.LINE_AA)
    return image


def draw_challenge(image, target, hold_count, score, flash_frames):
    """Draw the Challenge Mode overlay in the top-centre of the frame.

    Displays the target gesture name, a hold-progress bar, and the current
    score.  When *flash_frames* > 0 a green success flash is shown.

    Args:
        image: BGR frame to draw on (modified in-place).
        target: Target gesture label string.
        hold_count: Number of consecutive frames the target has been held.
        score: Current challenge score (gestures completed).
        flash_frames: Countdown frames remaining for the success flash.

    Returns:
        The annotated *image*.
    """
    h, w = image.shape[:2]
    panel_w, panel_h = 340, 72
    x0 = (w - panel_w) // 2
    y0 = 8

    if flash_frames > 0:
        alpha = min(flash_frames / _CHALLENGE_FLASH_FRAMES, 1.0) * 0.75
        _fill_semitransparent(image, x0, y0, x0 + panel_w, y0 + panel_h,
                              (0, 160, 0), alpha=alpha)
        cv.putText(image, "NICE!  +1", (x0 + panel_w // 2 - 60, y0 + 46),
                   cv.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2,
                   cv.LINE_AA)
    else:
        _fill_semitransparent(image, x0, y0, x0 + panel_w, y0 + panel_h,
                              (20, 20, 20), alpha=0.72)
        cv.putText(image, "CHALLENGE", (x0 + 8, y0 + 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv.LINE_AA)
        cv.putText(image, target, (x0 + 8, y0 + 46),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                   cv.LINE_AA)

        # Progress bar
        bar_x, bar_y = x0 + 8, y0 + 58
        bar_w = panel_w - 16
        bar_h = 8
        progress = min(hold_count / _CHALLENGE_HOLD_FRAMES, 1.0)
        cv.rectangle(image, (bar_x, bar_y),
                     (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        if progress > 0:
            fill_color = (
                0,
                int(140 + 80 * progress),
                int(255 * (1 - progress)),
            )
            cv.rectangle(image, (bar_x, bar_y),
                         (bar_x + int(bar_w * progress), bar_y + bar_h),
                         fill_color, -1)

    # Score badge (top-right of panel)
    cv.putText(image, f"Score: {score}", (x0 + panel_w - 95, y0 + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv.LINE_AA)

    return image


def draw_streak_banner(image, streak_count):
    """Flash a centred milestone banner when a streak threshold is reached.

    Args:
        image: BGR frame to draw on (modified in-place).
        streak_count: Current streak count displayed in the banner text.

    Returns:
        The annotated *image*.
    """
    h, w = image.shape[:2]
    text = f"x{streak_count} STREAK!"
    font = cv.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.4, 3
    (tw, th), _ = cv.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = h // 2
    _fill_semitransparent(image, x - 20, y - th - 16, x + tw + 20, y + 10,
                          (0, 80, 180), alpha=0.7)
    cv.putText(image, text, (x, y), font, scale, (0, 255, 200), thickness,
               cv.LINE_AA)
    return image


if __name__ == "__main__":
    main()
