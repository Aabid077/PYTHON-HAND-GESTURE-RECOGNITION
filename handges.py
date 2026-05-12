#HAND GESTURE PROJECTS

"""
Hand Gesture Detection using MediaPipe and OpenCV
==================================================
Detects and classifies hand gestures in real-time using webcam.
 
Gestures Supported:
  ✊ Fist        - All fingers closed
  ✋ Open Hand   - All fingers open
  👍 Thumbs Up   - Only thumb extended upward
  👎 Thumbs Down - Only thumb extended downward
  ✌️ Peace       - Index + Middle fingers open
  🤘 Rock On     - Index + Pinky fingers open
  👆 Pointing    - Only Index finger open
  🤙 Call Me     - Thumb + Pinky fingers open
 
Requirements:
    pip install opencv-python mediapipe
 
Usage:
    python hand_gesture_detection.py
 
Controls:
    Q / ESC - Quit
    S       - Save screenshot
    F       - Toggle FPS display
"""

import cv2
import mediapipe as mp
import math
import time
import os
from datetime import datetime
print ("ALL OKAY")


# ──────────────────────────────────────────────
#  Constants & Configuration
# ──────────────────────────────────────────────

GESTURES = {
    "FIST":        "✊ Fist",
    "OPEN_HAND":   "✋ Open Hand",
    "THUMBS_UP":   "👍 Thumbs Up",
    "THUMBS_DOWN": "👎 Thumbs Down",
    "PEACE":       "✌️ Peace",
    "ROCK_ON":     "🤘 Rock On",
    "POINTING":    "👆 Pointing",
    "CALL_ME":     "🤙 Call Me",
    "UNKNOWN":     "❓ Unknown",
}

# Landmark indices (MediaPipe hand model)
WRIST        = 0
THUMB_CMC    = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP    = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP   = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP     = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP    = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20


# ──────────────────────────────────────────────
#  Helper Functions
# ──────────────────────────────────────────────

def get_landmark_coords(landmarks, idx, w, h):
    """Return pixel coordinates for a landmark index."""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def euclidean(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_finger_extended(landmarks, tip_idx, pip_idx, mcp_idx):
    """
    Returns True if a finger is extended.
    Checks that TIP is further from WRIST than PIP,
    using y-coordinates (lower y = higher in frame).
    """
    tip_y = landmarks[tip_idx].y
    pip_y = landmarks[pip_idx].y
    mcp_y = landmarks[mcp_idx].y
    # Extended if TIP is above PIP by a margin
    return tip_y < pip_y - 0.02


def is_thumb_extended(landmarks, hand_label):
    """
    Special logic for thumb: compare x-coordinates
    accounting for left/right hand mirroring.
    """
    tip_x  = landmarks[THUMB_TIP].x
    ip_x   = landmarks[THUMB_IP].x
    mcp_x  = landmarks[THUMB_MCP].x

    if hand_label == "Right":
        return tip_x < ip_x - 0.02   # tip is to the left of ip
    else:
        return tip_x > ip_x + 0.02   # tip is to the right of ip


def get_finger_states(landmarks, hand_label):
    """
    Returns a dict of booleans for each finger.
    Keys: thumb, index, middle, ring, pinky
    """
    return {
        "thumb":  is_thumb_extended(landmarks, hand_label),
        "index":  is_finger_extended(landmarks, INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
        "middle": is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        "ring":   is_finger_extended(landmarks, RING_TIP,   RING_PIP,   RING_MCP),
        "pinky":  is_finger_extended(landmarks, PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
    }


def classify_gesture(landmarks, hand_label):
    """Classify gesture based on finger states and hand orientation."""
    f = get_finger_states(landmarks, hand_label)
    open_count = sum(f.values())

    wrist_y     = landmarks[WRIST].y
    thumb_tip_y = landmarks[THUMB_TIP].y

    # ── All closed → Fist
    if open_count == 0:
        return "FIST"

    # ── All open → Open Hand
    if open_count == 5:
        return "OPEN_HAND"

    # ── Only thumb open
    if f["thumb"] and not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]:
        if thumb_tip_y < wrist_y:   # thumb points upward
            return "THUMBS_UP"
        else:
            return "THUMBS_DOWN"

    # ── Index + Middle (Peace / V-sign)
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"] and not f["thumb"]:
        return "PEACE"

    # ── Index + Pinky (Rock On / Horns)
    if f["index"] and f["pinky"] and not f["middle"] and not f["ring"]:
        return "ROCK_ON"

    # ── Only Index (Pointing)
    if f["index"] and not f["middle"] and not f["ring"] and not f["pinky"] and not f["thumb"]:
        return "POINTING"

    # ── Thumb + Pinky (Call Me / Shaka)
    if f["thumb"] and f["pinky"] and not f["index"] and not f["middle"] and not f["ring"]:
        return "CALL_ME"

    return "UNKNOWN"


# ──────────────────────────────────────────────
#  Drawing Utilities
# ──────────────────────────────────────────────

COLOR_PRIMARY   = (0, 220, 150)     # green-cyan
COLOR_SECONDARY = (255, 200, 50)    # amber
COLOR_BG        = (15, 15, 25)      # near-black
COLOR_TEXT      = (240, 240, 240)   # white
COLOR_ALERT     = (60, 120, 255)    # blue


def draw_info_panel(frame, gesture_key, hand_label, fps, show_fps):
    """Draws a semi-transparent HUD overlay with gesture info."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Background panel
    cv2.rectangle(overlay, (10, 10), (370, 110), (20, 20, 35), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Border
    cv2.rectangle(frame, (10, 10), (370, 110), COLOR_PRIMARY, 2)

    # Gesture label
    gesture_text = GESTURES.get(gesture_key, GESTURES["UNKNOWN"])
    cv2.putText(frame, gesture_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_SECONDARY, 3, cv2.LINE_AA)

    # Hand side
    side_text = f"{hand_label} Hand"
    cv2.putText(frame, side_text, (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

    # FPS
    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 130, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PRIMARY, 2, cv2.LINE_AA)


def draw_controls(frame):
    """Draws key-binding hints at the bottom of the frame."""
    h, w = frame.shape[:2]
    hints = "Q/ESC: Quit   |   S: Screenshot   |   F: Toggle FPS"
    cv2.putText(frame, hints, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 160), 1, cv2.LINE_AA)


def draw_landmark_connections(frame, landmarks, h, w):
    """Draws colored finger skeleton over hand."""
    finger_colors = [
        (COLOR_SECONDARY, [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP]),
        (COLOR_PRIMARY,   [INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP]),
        ((50, 200, 255),  [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP]),
        ((200, 100, 255), [RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP]),
        ((255, 120, 80),  [PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP]),
    ]

    # Palm connections
    palm = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    pts  = [get_landmark_coords(landmarks, i, w, h) for i in palm]
    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i + 1], (100, 100, 130), 2)
    cv2.line(frame, pts[0], pts[1], (100, 100, 130), 2)
    cv2.line(frame, pts[0], pts[-1], (100, 100, 130), 2)

    # Finger connections
    for color, chain in finger_colors:
        for i in range(len(chain) - 1):
            p1 = get_landmark_coords(landmarks, chain[i],     w, h)
            p2 = get_landmark_coords(landmarks, chain[i + 1], w, h)
            cv2.line(frame, p1, p2, color, 3)

    # Joint dots
    for idx in range(21):
        pt = get_landmark_coords(landmarks, idx, w, h)
        cv2.circle(frame, pt, 5, COLOR_TEXT, -1)
        cv2.circle(frame, pt, 5, (50, 50, 80), 1)


# ──────────────────────────────────────────────
#  Main Application
# ──────────────────────────────────────────────

def main():
    # MediaPipe setup
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that a camera is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 52)
    print("  Hand Gesture Detection — Running")
    print("  Q / ESC : Quit")
    print("  S       : Save screenshot")
    print("  F       : Toggle FPS display")
    print("=" * 52)

    prev_time  = time.time()
    fps        = 0.0
    show_fps   = True
    screenshot_dir = "screenshots"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame capture failed. Retrying…")
            continue

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        # ── FPS Calculation
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ── Process detected hands
        if results.multi_hand_landmarks:
            for hand_lm, hand_info in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = hand_info.classification[0].label  # "Left" or "Right"
                landmarks  = hand_lm.landmark

                # Custom skeleton drawing
                draw_landmark_connections(frame, landmarks, h, w)

                # Gesture classification
                gesture_key = classify_gesture(landmarks, hand_label)

                # HUD overlay
                draw_info_panel(frame, gesture_key, hand_label, fps, show_fps)

        else:
            # No hand detected
            cv2.putText(frame, "No Hand Detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 120), 2, cv2.LINE_AA)

        draw_controls(frame)

        cv2.imshow("Hand Gesture Detection", frame)

        # ── Key Handling
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):       # Q or ESC → quit
            break

        elif key == ord('s'):           # S → screenshot
            os.makedirs(screenshot_dir, exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshot_dir, f"gesture_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot saved: {path}")

        elif key == ord('f'):           # F → toggle FPS
            show_fps = not show_fps

    # Cleanup
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")


if __name__ == "__main__":
    main()

    print("Successful")