import cv2
import mediapipe as mp
import numpy as np

class SkeletonExtractor:
    def __init__(self):
        # We instantiate this inside the worker processes to avoid pickling errors
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.5
        )

    def extract_from_frame(self, image_path):
        """Extracts 21 (x,y,z) coordinates. Returns None if hand not found."""
        image = cv2.imread(image_path)
        if image is None: 
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image_rgb)
        
        if not results.multi_hand_landmarks: 
            return None
        
        coords = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
        return np.array(coords, dtype=np.float32) # Shape: (21, 3)

    def close(self):
        self.mp_hands.close()