import mediapipe as mp
import cv2
import numpy as np


class HandTracker:
    def __init__(self, frame_width=640):
        self.frame_width = frame_width
        self.bucket_x = frame_width // 2
        self.prev_bucket_x = self.bucket_x

        # For gesture detection
        self.gesture = "none"

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.mp_draw = mp.solutions.drawing_utils

    # Measure distance between two landmarks
    def dist(self, lm, p1, p2):
        x1, y1 = lm[p1].x, lm[p1].y
        x2, y2 = lm[p2].x, lm[p2].y
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Detect which gesture is shown
    def detect_gesture(self, lm):
        thumb_index = self.dist(lm, 4, 8)
        thumb_pinky = self.dist(lm, 4, 20)
        spread = self.dist(lm, 5, 17)

        # ---- ✋ OPEN PALM (START GAME) ----
        if spread > 0.35:
            return "start"

        # ---- 🤏 PINCH (PAUSE GAME) ----
        if thumb_index < 0.07:
            return "pause"

        # ---- 🤙 SHAKA (RESTART) ----
        if thumb_index > 0.2 and thumb_pinky > 0.25:
            return "restart"

        return "none"

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            # ----- BUCKET POSITION (Thumb + Index midpoint) -----
            thumb_x = int(lm[4].x * self.frame_width)
            index_x = int(lm[8].x * self.frame_width)
            target_x = (thumb_x + index_x) // 2

            # Smooth the bucket movement (anti-shaking)
            self.bucket_x = int(self.prev_bucket_x * 0.7 + target_x * 0.3)
            self.prev_bucket_x = self.bucket_x

            # ------ Detect Gesture ------
            self.gesture = self.detect_gesture(lm)

        return frame

    # These values will be used by Flask (/bucket_position)
    def get_bucket_info(self):
        return {
            "x": int(self.bucket_x),
            "gesture": self.gesture
        }
