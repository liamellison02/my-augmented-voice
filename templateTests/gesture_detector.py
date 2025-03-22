import csv
import ast
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import mediapipe as mp
import cv2 

def load_time_series_from_csv(filepath):
    time_steps = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Each row is a time step; parse each cell
            flattened = []
            for cell in row:
                # Safely evaluate the tuple string into a Python tuple
                try:
                    vector = ast.literal_eval(cell)
                except Exception as e:
                    raise ValueError(f"Error parsing cell '{cell}': {e}")
                # If the evaluated cell is a tuple or list, extend; otherwise, add as single float
                if isinstance(vector, (list, tuple)):
                    flattened.extend(vector)
                else:
                    flattened.append(vector)
            time_steps.append(flattened)
    
    return np.array(time_steps)

def func_init(): 
    csv_files = [
    "output_csv/howareyou.csv", 
    "output_csv/nicetomeetu.csv", 
    ]
    arr = []
    for csv_file in csv_files: 
        D_entry = load_time_series_from_csv(csv_file)
        arr.append(D_entry)
    return arr 

def best_match_dtw(V, D):
    best_index = None
    best_distance = float('inf')
    for i, series in enumerate(D):
        distance, _ = fastdtw(V, series, dist=euclidean)
        print(distance)
        if distance < best_distance:
            best_distance = distance
            best_index = i
    return best_index
    
if __name__ == '__main__':
    templates = func_init()
    gesture_labels = ["howareyou", "nicetomeetu"]
    model_path = r"C:\Users\joelj\projects\my-augmented-voice\templateTests\model\hand_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

    results = ""

    cap = cv2.VideoCapture(0); 
    gesture_frames = []      
    capturing = True        
    capture_length = 50  
    last_recog_phrase = ""

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)

            landmark_vector = []
            if results.hand_world_landmarks: 
                for hand_index, hand_world_landmarks in enumerate(results.hand_world_landmarks):
                    for landmark_index, landmark in enumerate(hand_world_landmarks):
                        landmark_vector.extend([landmark.x, landmark.y, landmark.z])
                
            
                if capturing:
                    gesture_frames.append(landmark_vector)
            
            cv2.putText(frame, "'s' to start gesture capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.putText(frame, last_recog_phrase, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if capturing:
                cv2.putText(frame, f"Capturing: {len(gesture_frames)}/{capture_length}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # If we have captured enough frames, perform matching.
            if capturing and len(gesture_frames) >= capture_length:
                V = np.array(gesture_frames)
                match_index = best_match_dtw(V, templates)
                gesture_name = gesture_labels[match_index] if match_index is not None else "Unknown"

                last_recog_phrase = f"Gesture: {gesture_name}" 
                                
                capturing = False
                gesture_frames = []

            cv2.imshow("Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            # Press 's' to start capturing a new gesture.
            if key == ord('s') and not capturing:
                capturing = True
                gesture_frames = []
            # Press 'q' to quit.
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows() 
 
