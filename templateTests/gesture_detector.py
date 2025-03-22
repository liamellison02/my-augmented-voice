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
        time_steps = list(reader)

    print(len(time_steps[0]))

    return np.array(time_steps)

def func_init(): 
    csv_files = [
    "output_csv/howareyou.csv", 
    "output_csv/nicetomeetu.csv", 
    "output_csv/mynameis.csv", 
    ]
    arr = []
    for csv_file in csv_files: 
        D_entry = load_time_series_from_csv(csv_file)
        arr.append(D_entry)
    return arr 

def get_important_nodes(x): 
    mask = [False] * (21 * 3  * 2)  # 21 landmarks * 3 coordinates (x, y, z)
    values = list(range(8 + 1)) #Thumb Index and Palm nodes 

    for i in values: 
       mask[i*3] = True
       mask[i*3 + 1] = True
       mask[i*3 + 2] = True

       mask[(i+21)*3] = True
       mask[(i+21)*3 + 1] = True
       mask[(i+21)*3 + 2] = True

    x = np.array(x)
    x = x[mask]

    return x  


def best_match_dtw(V, D, dist_func):
    best_index = None
    best_distance = float('inf')
    for i, series in enumerate(D):
        print(len(series), len(V))
        distance, path= fastdtw(V, series, dist=dist_func)
        normalized_distance = distance/len(path[0]) 
        print(normalized_distance)
        if normalized_distance < best_distance:
            best_distance = normalized_distance
            best_index = i
    print("")

    return best_index

def manhattan_distance(x, y):
    # x = np.array(x)
    # y = np.array(y)
    # mask = (x != -99) & (y != -99)
    
    # if np.sum(mask) == 0:
    #     return 0.0

    # x = x[mask]
    # y = y[mask]

    return sum(abs(a - b) for a, b in zip(x, y))

def chebyshev_distance(x, y):
    # x = np.array(x)
    # y = np.array(y)
    # mask = (x != -99) & (y != -99)
    
    # if np.sum(mask) == 0:
    #     return 0.0
    # x = x[mask]
    # y = y[mask]
    return max(abs(a - b) for a, b in zip(x, y))

def minkowski_distance(x, y, p=3):  # try different p values
    # x = np.array(x)
    # y = np.array(y)
    # mask = (x != -99) & (y != -99)
    
    # if np.sum(mask) == 0:
    #     return 0.0
    # x = x[mask]
    # y = y[mask]
 
    return sum(abs(a - b) ** p for a, b in zip(x, y)) ** (1 / p)
 
if __name__ == '__main__':
    dist_algo = [
        euclidean, 
        manhattan_distance, 
        chebyshev_distance, 
        minkowski_distance
    ]

    templates = func_init()
    gesture_labels = ["howareyou", "nicetomeetu", "my name is "]
    model_path = r"C:\Users\joelj\projects\my-augmented-voice\templateTests\model\hand_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, num_hands=2)

    results = ""

    cap = cv2.VideoCapture(0); 
    gesture_frames = []      
    capturing = True        
    capture_length = 267  
    last_recog_phrase = ""
    multi_recog_phrases =[] 

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)
            row_data = []  # Start with the frame number

                # Initialize left and right hand data lists
            left_hand_data = []
            right_hand_data = []

            if results.hand_world_landmarks:
                for hand_index, hand_world_landmarks in enumerate(results.hand_world_landmarks):
                    hand_data = []
                    for landmark_index, landmark in enumerate(hand_world_landmarks):
                        hand_data.extend([landmark.x, landmark.y, landmark.z])  # Add x, y, z coordinates


                    # Assuming handedness is reliable, assign to left or right hand
                    if results.handedness and len(results.handedness) > hand_index:
                        handedness = results.handedness[hand_index][0].category_name  # Get "Left" or "Right"
                        if handedness == "Left":
                            left_hand_data = hand_data
                        elif handedness == "Right":
                            right_hand_data = hand_data
                        else:
                            print(f"Unexpected handedness: {handedness}") # Debugging
                    else:
                        print("Handedness information not available or not enough hands detected") # Debugging

                padding = [0] * (21 * 3)  # 21 landmarks * 3 coordinates (x, y, z)
                if not left_hand_data:
                    left_hand_data = padding
                if not right_hand_data:
                    right_hand_data = padding

                row_data.extend(left_hand_data)
                row_data.extend(right_hand_data)

                # landmark_vector = []
                # if results.hand_world_landmarks: 
                #     for hand_index, hand_world_landmarks in enumerate(results.hand_world_landmarks):
                #         for landmark_index, landmark in enumerate(hand_world_landmarks):
                #             landmark_vector.extend([landmark.x, landmark.y, landmark.z])
                
                if capturing:
                    gesture_frames.append(row_data)
                
            cv2.putText(frame, "'s' to start gesture capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.putText(frame, last_recog_phrase, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            for index, phrase in enumerate(multi_recog_phrases): 
                cv2.putText(frame, phrase, (400, 90-(index*20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if capturing:
                cv2.putText(frame, f"Capturing: {len(gesture_frames)}/{capture_length}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # If we have captured enough frames, perform matching.
            if capturing and len(gesture_frames) >= capture_length:
                V = np.array(gesture_frames)
                match_index = best_match_dtw(V, templates, euclidean)
                for dist_func in dist_algo: 
                    new_detected = best_match_dtw(V, templates, dist_func)
                    if(new_detected is not None): 
                        multi_recog_phrases.append(f"{gesture_labels[new_detected]}")
                    else: 
                        multi_recog_phrases.append("")
                        
                gesture_name = gesture_labels[match_index] if match_index is not None else "Unknown"

                last_recog_phrase = f"Gesture: {gesture_name}" 
                                
                capturing = False
                gesture_frames = []

            cv2.imshow("Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            # Press 's' to start capturing a new gesture.
            if key == ord('s') and not capturing:
                capturing = True
                multi_recog_phrases = []
                gesture_frames = []
            # Press 'q' to quit.
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows() 
 
