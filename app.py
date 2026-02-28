#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Hand Gesture Recognition application.

Real-time hand sign and finger gesture classifier using MediaPipe Hands,
OpenCV, and trained TensorFlow/Keras models.  Run with::

    python app.py [--device 0] [--width 960] [--height 540]

Press ``k`` to enter keypoint-logging mode, ``h`` for point-history-logging
mode, and ``ESC`` to quit.
"""

import csv
import argparse
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier import PointHistoryClassifier

# ---------------------------------------------------------------------------
# Hand skeleton: pairs of landmark indices that form a bone segment.
# Each entry is drawn twice — once thick black (border) then thin white (fill).
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

    parser.add_argument('--use_static_image_mode', action='store_true',
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
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    # FPS counter and history buffers
    cv_fps_calc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    while True:
        fps = cv_fps_calc.get()

        # Process key input (ESC = quit)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = image.copy()

        # Run MediaPipe hand detection on RGB frame
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
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
                if hand_sign_id == "Not Applicable":  # point gesture
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

                # Overlay drawing
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    """Map a key-press to a (number, mode) pair.

    Args:
        key: Integer key code from ``cv.waitKey``.
        mode: Current operating mode (0 = normal, 1 = keypoint log,
              2 = point-history log).

    Returns:
        Tuple of (number, mode) where *number* is 0–9 if a digit key was
        pressed, otherwise -1.
    """
    number = -1
    if 48 <= key <= 57:  # 0–9
        number = key - 48
    if key == 110:    # n — normal mode
        mode = 0
    elif key == 107:  # k — keypoint logging
        mode = 1
    elif key == 104:  # h — point-history logging
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    """Compute the axis-aligned bounding rectangle for a set of hand landmarks.

    Uses a list comprehension to build the landmark array in one allocation,
    avoiding the O(n²) cost of repeated ``np.append`` calls in a loop.

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
    NumPy vectorised operations replace the previous Python loops.

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
    NumPy vectorised operations replace the previous Python loops.

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
        number: Digit label (0–9) pressed by the user, or -1 for none.
        mode: Current mode (0 = normal, 1 = keypoint log,
              2 = point-history log).
        landmark_list: Pre-processed landmark feature vector.
        point_history_list: Pre-processed point-history feature vector.
    """
    if mode == 1 and 0 <= number <= 9:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])
    elif mode == 2 and 0 <= number <= 9:
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            csv.writer(f).writerow([number, *point_history_list])


def draw_landmarks(image, landmark_point):
    """Draw hand skeleton and keypoints onto *image*.

    Bone segments are drawn using the connection list ``_HAND_CONNECTIONS``
    (each rendered twice: thick black border then thin white fill).  Landmark
    circles are sized by fingertip membership rather than 21 separate ``if``
    branches.

    Args:
        image: BGR frame to draw on (modified in-place).
        landmark_point: List of ``[x, y]`` pixel coordinates (21 points).

    Returns:
        The annotated *image*.
    """
    if not landmark_point:
        return image

    # Draw bone connections
    for start, end in _HAND_CONNECTIONS:
        pt1 = tuple(landmark_point[start])
        pt2 = tuple(landmark_point[end])
        cv.line(image, pt1, pt2, (0, 0, 0), 6)
        cv.line(image, pt1, pt2, (255, 255, 255), 2)

    # Draw landmark circles (fingertips are larger)
    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in _FINGERTIP_INDICES else 5
        center = (landmark[0], landmark[1])
        cv.circle(image, center, radius, (255, 255, 255), -1)
        cv.circle(image, center, radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    """Draw an axis-aligned bounding rectangle around the detected hand.

    Args:
        use_brect: If ``False`` the rectangle is skipped.
        image: BGR frame to draw on (modified in-place).
        brect: ``[x1, y1, x2, y2]`` pixel coordinates.

    Returns:
        The annotated *image*.
    """
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    """Overlay handedness and classification label above the bounding box.

    Args:
        image: BGR frame to draw on (modified in-place).
        brect: ``[x1, y1, x2, y2]`` pixel coordinates.
        handedness: MediaPipe ``ClassificationList`` for the detected hand.
        hand_sign_text: Label string from the keypoint classifier.
        finger_gesture_text: Label string from the point-history classifier
            (currently reserved for future use).

    Returns:
        The annotated *image*.
    """
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label
    if hand_sign_text:
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    """Draw the finger-tip trajectory as a trail of growing circles.

    Args:
        image: BGR frame to draw on (modified in-place).
        point_history: Deque of ``[x, y]`` pixel coordinates.

    Returns:
        The annotated *image*.
    """
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    """Overlay FPS counter and current mode/number onto *image*.

    Args:
        image: BGR frame to draw on (modified in-place).
        fps: Current frames-per-second value.
        mode: Current operating mode integer.
        number: Currently selected digit label (0–9), or -1.

    Returns:
        The annotated *image*.
    """
    cv.putText(image, f"FPS:{fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, f"FPS:{fps}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, f"MODE:{mode_string[mode - 1]}", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, f"NUM:{number}", (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
