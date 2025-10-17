import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
import time
import math
from PIL import Image

WEBCAM_ID = 0
WINDOW_NAME = "Clash Royale Gesture Emotes"
UI_WIDTH = 1280
UI_HEIGHT = 720
WEBCAM_FEED_WIDTH = UI_WIDTH // 2
EMOTE_DISPLAY_WIDTH = UI_WIDTH // 2
BG_COLOR = (19, 21, 23)

GESTURE_COOLDOWN = 1.5 # Seconds
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
CONFIRMATION_FRAMES = 4 

EMOTE_DIR = "emotes"

GESTURE_TO_EMOTE = {
    "neutral": "neutral.png",
    "thumbs_up": "thumbs_up.gif",
    "cry": "goblin_crying.gif",
    "kiss": "princess_kiss.gif",
    "laugh": "king_laugh.gif",
    "angry": "angry_king.gif",
    "mog": "mog.gif",
}

def load_emotes(emote_map, directory):
    emotes = {}
    for gesture, filename in emote_map.items():
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            print(f"Warning: Emote image not found at {path}")
            placeholder = np.zeros((200, 200, 3), dtype=np.uint8)
            emotes[gesture] = {'frames': [placeholder], 'type': 'png'}
            continue
        if filename.endswith(".gif"):
            pil_gif = Image.open(path)
            frames = []
            for i in range(pil_gif.n_frames):
                pil_gif.seek(i)
                frame_rgba = pil_gif.convert("RGBA")
                frame_cv = cv2.cvtColor(np.array(frame_rgba), cv2.COLOR_RGBA2BGRA)
                frames.append(frame_cv)
            emotes[gesture] = {'frames': frames, 'frame_count': len(frames), 'current_idx': 0, 'last_update': 0, 'type': 'gif'}
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            emotes[gesture] = {'frames': [img], 'type': 'png'}
    return emotes

class GestureRecognizer:
    def __init__(self):
        self.last_gesture_time = 0
        self.current_gesture = "neutral"
        self.debug_info = {}
        self.potential_gesture = "neutral"
        self.confirmation_counter = 0

    def get_landmark_coords(self, landmarks, landmark_index, frame_shape):
        lm = landmarks.landmark[landmark_index]
        return int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def recognize(self, multi_hand_landmarks, multi_face_landmarks, frame_shape):
        self.debug_info = {}
        detected_gesture = "neutral"

        if time.time() - self.last_gesture_time < GESTURE_COOLDOWN:
            return self.current_gesture

        face_landmarks = multi_face_landmarks[0] if multi_face_landmarks else None

        if multi_hand_landmarks and face_landmarks:
            # 1. Cry
            if len(multi_hand_landmarks) == 2:
                hand1_palm = self.get_landmark_coords(multi_hand_landmarks[0], 9, frame_shape)
                hand2_palm = self.get_landmark_coords(multi_hand_landmarks[1], 9, frame_shape)
                left_eye = self.get_landmark_coords(face_landmarks, 133, frame_shape)
                right_eye = self.get_landmark_coords(face_landmarks, 362, frame_shape)
                
                dist1 = self.calculate_distance(hand1_palm, right_eye)
                dist2 = self.calculate_distance(hand2_palm, left_eye)
                self.debug_info['EyeDist'] = f"L:{dist2:.0f}, R:{dist1:.0f}"
                if dist1 < 120 and dist2 < 120:
                    detected_gesture = "cry"
                    
            elif len(multi_hand_landmarks) == 1:
                hand_landmarks = multi_hand_landmarks[0]
                
                # Kiss 
                palm_center = self.get_landmark_coords(hand_landmarks, 9, frame_shape)
                lips_top = self.get_landmark_coords(face_landmarks, 13, frame_shape)
                dist_to_lips = self.calculate_distance(palm_center, lips_top)
                self.debug_info['LipDist'] = f"{dist_to_lips:.0f}"
                if dist_to_lips < 80:
                     detected_gesture = "kiss"
                
                # Mog
                else:
                    index_tip = self.get_landmark_coords(hand_landmarks, 8, frame_shape)
                    chin = self.get_landmark_coords(face_landmarks, 152, frame_shape)
                    dist_to_chin = self.calculate_distance(index_tip, chin)
                    self.debug_info['ChinDist'] = f"{dist_to_chin:.0f}"
                    index_mcp = self.get_landmark_coords(hand_landmarks, 5, frame_shape)
                    middle_tip = self.get_landmark_coords(hand_landmarks, 12, frame_shape)
                    middle_mcp = self.get_landmark_coords(hand_landmarks, 9, frame_shape)
                    is_index_up = index_tip[1] < index_mcp[1] and middle_tip[1] > middle_mcp[1]
                    
                    if dist_to_chin < 80 and is_index_up:
                        detected_gesture = "mog"

        # Thumbs Up 
        if detected_gesture == "neutral" and multi_hand_landmarks and len(multi_hand_landmarks) == 1:
            hand_landmarks = multi_hand_landmarks[0]
            thumb_tip = self.get_landmark_coords(hand_landmarks, 4, frame_shape)
            index_knuckle = self.get_landmark_coords(hand_landmarks, 5, frame_shape)
            if thumb_tip[1] < index_knuckle[1]:
                fingers_down = all(
                    self.get_landmark_coords(hand_landmarks, i, frame_shape)[1] > self.get_landmark_coords(hand_landmarks, i - 2, frame_shape)[1]
                    for i in [8, 12, 16, 20]
                )
                if fingers_down:
                    detected_gesture = "thumbs_up"

        if detected_gesture == "neutral" and face_landmarks:
            nose_tip = self.get_landmark_coords(face_landmarks, 1, frame_shape)
            nose_bridge = self.get_landmark_coords(face_landmarks, 6, frame_shape)
            vertical_nose_dist = nose_tip[1] - nose_bridge[1]
            self.debug_info['NoseTilt'] = f"{vertical_nose_dist}"
            
            # Heheheha 
            if vertical_nose_dist < 25:
                detected_gesture = "laugh"
            # Angry King 
            elif vertical_nose_dist > 45:
                detected_gesture = "angry"

        # --- Machine Logic ---
        if detected_gesture == self.potential_gesture:
            self.confirmation_counter += 1
        else:
            self.potential_gesture = detected_gesture
            self.confirmation_counter = 0

        self.debug_info['Status'] = f"Confirming: {self.potential_gesture} ({self.confirmation_counter}/{CONFIRMATION_FRAMES})"

        if self.confirmation_counter >= CONFIRMATION_FRAMES:
            if self.potential_gesture != self.current_gesture:
                self.current_gesture = self.potential_gesture
                self.last_gesture_time = time.time()
            if self.potential_gesture == "neutral":
                 self.current_gesture = "neutral"
            self.confirmation_counter = 0
        
        if not multi_hand_landmarks and not face_landmarks:
            self.current_gesture = "neutral"

        return self.current_gesture

def draw_debug_info(frame, recognizer):
    y_pos = 30
    for key, value in recognizer.debug_info.items():
        text = f"{key}: {value}"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y_pos += 30

def main():
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE, max_num_hands=2)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1)

    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotes = load_emotes(GESTURE_TO_EMOTE, EMOTE_DIR)
    recognizer = GestureRecognizer()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_hands = hands.process(rgb_frame)
        results_face = face_mesh.process(rgb_frame)
        
        frame.flags.writeable = True

        current_gesture = recognizer.recognize(results_hands.multi_hand_landmarks, results_face.multi_face_landmarks, frame.shape)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                 mp_drawing.draw_landmarks(
                    image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        draw_debug_info(frame, recognizer)

        # UI 
        ui_canvas = np.full((UI_HEIGHT, UI_WIDTH, 3), BG_COLOR, dtype=np.uint8)
        webcam_feed = cv2.resize(frame, (WEBCAM_FEED_WIDTH, UI_HEIGHT))
        ui_canvas[:, 0:WEBCAM_FEED_WIDTH] = webcam_feed

        emote_data = emotes.get(current_gesture, emotes["neutral"])
        
        if emote_data['type'] == 'gif':
            if (time.time() - emote_data['last_update']) > 0.1:
                emote_data['current_idx'] = (emote_data['current_idx'] + 1) % emote_data['frame_count']
                emote_data['last_update'] = time.time()
            emote_img = emote_data['frames'][emote_data['current_idx']]
        else:
            emote_img = emote_data['frames'][0]
        
        h, w = emote_img.shape[:2]
        if emote_img.shape[2] == 4:
            alpha = emote_img[:, :, 3] / 255.0
            bg_panel = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
            blended = (emote_img[:, :, :3] * alpha[..., np.newaxis] + bg_panel * (1 - alpha[..., np.newaxis])).astype(np.uint8)
        else:
            blended = emote_img
        
        scale = min(EMOTE_DISPLAY_WIDTH / w, UI_HEIGHT / h) * 0.9
        tw, th = int(w * scale), int(h * scale)
        resized = cv2.resize(blended, (tw, th))
        
        y_off = (UI_HEIGHT - th) // 2
        x_off = WEBCAM_FEED_WIDTH + (EMOTE_DISPLAY_WIDTH - tw) // 2
        ui_canvas[y_off:y_off+th, x_off:x_off+tw] = resized
        
        cv2.putText(ui_canvas, f"Gesture: {current_gesture.upper()}", (WEBCAM_FEED_WIDTH + 20, 50), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, ui_canvas)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()

if __name__ == "__main__":
    main()

