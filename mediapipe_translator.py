"""
ASL Translator using MediaPipe Hand detection
"""
import cv2
import numpy as np
import time
import pyttsx3
import mediapipe as mp

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ASLTranslator:
    def __init__(self):
        self.signs = {
            "HELLO": self.detect_hello,
            "THANK YOU": self.detect_thank_you,
            "YES": self.detect_yes,
            "NO": self.detect_no
        }
        
        # For movement tracking
        self.prev_landmarks = None
        self.movement_history = []
        self.max_history = 10
        
        # For output stability
        self.last_sign = None
        self.sign_count = 0
        self.min_detection_count = 3
        self.last_spoken_time = 0
        self.cooldown = 2  # seconds between speaking the same sign
    
    def detect_hello(self, hand_landmarks, hand_type):
        """
        HELLO sign - open hand with fingers spread
        """
        if hand_type != "Right":  # Assuming right hand for all signs
            return False
            
        # Get fingertips and check if they're extended
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        
        # Check if all fingers are extended
        thumb_extended = thumb_tip.x > wrist.x  # For right hand
        index_extended = index_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_extended = middle_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_extended = ring_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_extended = pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
        
        if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            # Check palm orientation (should be facing camera)
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # For right hand, Z value of MCP should be less than wrist for palm facing camera
            palm_facing = middle_mcp.z < wrist.z
            
            return palm_facing
        
        return False
    
    def detect_thank_you(self, hand_landmarks, hand_type):
        """
        THANK YOU sign - Flat hand moving from chin outward
        """
        if hand_type != "Right":
            return False
            
        # Check if hand is flat (fingers together)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        
        # Check if fingers are close together
        fingers_together = (
            abs(index_tip.y - middle_tip.y) < 0.05 and
            abs(middle_tip.y - ring_tip.y) < 0.05 and
            abs(ring_tip.y - pinky_tip.y) < 0.05
        )
        
        # Check forward movement
        if self.prev_landmarks is not None and fingers_together:
            prev_wrist = self.prev_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            curr_wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # Forward movement: z decreasing
            movement_z = prev_wrist.z - curr_wrist.z
            
            if movement_z > 0.01:  # Forward movement threshold
                return True
        
        return False
    
    def detect_yes(self, hand_landmarks, hand_type):
        """
        YES sign - Hand in fist shape with up and down movement
        """
        if hand_type != "Right":
            return False
            
        # Check for fist shape
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # Fingers should be curled in
        fingers_curled = (
            index_tip.y > index_pip.y and
            middle_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        )
        
        # Track vertical movement
        if self.prev_landmarks is not None and fingers_curled:
            prev_wrist = self.prev_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            curr_wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            
            # Track y movement (up/down)
            movement_y = curr_wrist.y - prev_wrist.y
            self.movement_history.append(movement_y)
            
            if len(self.movement_history) > self.max_history:
                self.movement_history.pop(0)
            
            if len(self.movement_history) >= 4:
                # Check for direction changes in y movement
                directions = [1 if move > 0 else -1 for move in self.movement_history]
                direction_changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
                
                if direction_changes >= 2:
                    self.movement_history = []
                    return True
        
        return False
    
    def detect_no(self, hand_landmarks, hand_type):
        """
        NO sign - Index finger extended, with side-to-side movement
        """
        if hand_type != "Right":
            return False
            
        # Check for index finger extended, others curled
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        
        index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        # Index extended, others curled
        index_extended = index_tip.y < index_pip.y
        others_curled = (
            middle_tip.y > middle_pip.y and
            ring_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
            pinky_tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
        )
        
        # Track horizontal movement
        if self.prev_landmarks is not None and index_extended and others_curled:
            prev_index = self.prev_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            curr_index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Track x movement (left/right)
            movement_x = curr_index.x - prev_index.x
            self.movement_history.append(movement_x)
            
            if len(self.movement_history) > self.max_history:
                self.movement_history.pop(0)
            
            if len(self.movement_history) >= 4:
                # Check for direction changes in x movement
                directions = [1 if move > 0 else -1 for move in self.movement_history]
                direction_changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
                
                if direction_changes >= 2:
                    self.movement_history = []
                    return True
        
        return False
    
    def recognize_sign(self, hand_landmarks, hand_type):
        """
        Recognize ASL sign based on hand landmarks.
        """
        for sign_name, detector in self.signs.items():
            if detector(hand_landmarks, hand_type):
                self.movement_history = []  # Reset movement history after detection
                return sign_name
        
        return None
    
    def process_frame(self, frame, results, speak_function):
        """
        Process frame with hand landmarks to detect ASL signs.
        """
        recognized_sign = None
        
        # Check if hands were detected
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand type (left or right)
                hand_type = results.multi_handedness[hand_idx].classification[0].label
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Recognize sign
                sign = self.recognize_sign(hand_landmarks, hand_type)
                if sign:
                    recognized_sign = sign
                
                # Update previous landmarks for movement tracking
                self.prev_landmarks = hand_landmarks
        else:
            # Reset movement tracking when no hands detected
            self.prev_landmarks = None
            self.movement_history = []
        
        # Handle sign recognition stability
        if recognized_sign:
            if recognized_sign == self.last_sign:
                self.sign_count += 1
            else:
                self.last_sign = recognized_sign
                self.sign_count = 1
            
            current_time = time.time()
            if (self.sign_count >= self.min_detection_count and 
                current_time - self.last_spoken_time > self.cooldown):
                # Display sign on frame
                cv2.putText(frame, f"TRANSLATION: {recognized_sign}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add a background for the translation text
                text_size = cv2.getTextSize(f"TRANSLATION: {recognized_sign}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(frame, (45, 20), (55 + text_size[0], 60), (0, 0, 0), -1)
                cv2.putText(frame, f"TRANSLATION: {recognized_sign}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Speak the sign
                speak_function(recognized_sign)
                self.last_spoken_time = current_time
        else:
            self.sign_count = 0
            # Still show the last recognized sign with a "Last detected:" prefix
            if self.last_sign:
                cv2.putText(frame, f"Last detected: {self.last_sign}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        
        # Always show status indicator for the translator
        cv2.putText(frame, "ASL Translator [ACTIVE]", (50, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                   
        return frame


def setup_camera():
    """
    Set up video capture.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap


def setup_tts():
    """
    Set up text-to-speech engine.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine


def speak(engine, text):
    """
    Speak text using TTS engine.
    """
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()


def main():
    """
    Main function for ASL translator.
    """
    print("Starting ASL Translator with MediaPipe...")
    
    # Setup camera
    cap = setup_camera()
    if cap is None:
        return
    
    # Setup TTS engine
    tts_engine = setup_tts()
    
    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
        
        # Initialize ASL translator
        translator = ASLTranslator()
        
        # For FPS calculation
        prev_time = time.time()
        frame_count = 0
        fps = 0
        
        print("Press 'q' to quit")
        print("Recognized signs: HELLO, THANK YOU, YES, NO")
        print("Instructions:")
        print("- HELLO: Show open palm with fingers spread")
        print("- THANK YOU: Move flat hand forward from mouth")
        print("- YES: Make a fist and nod up and down")
        print("- NO: Extend index finger and move side to side")
        
        while cap.isOpened():
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = hands.process(frame_rgb)
            
            # Process frame for ASL sign detection
            frame = translator.process_frame(frame, results, 
                                           lambda text: speak(tts_engine, text))
            
            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow('ASL Translator', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
