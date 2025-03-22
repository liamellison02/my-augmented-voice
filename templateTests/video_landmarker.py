import cv2
import csv
import argparse
from pprint import pprint
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def process_video(video_path, output_csv):

    model_path = r"C:\Users\joelj\projects\my-augmented-voice\templateTests\model\hand_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE, num_hands =2) 

    results = ""
    frame_number = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            input_video = cv2.VideoCapture(video_path)
            hand_land_marked_positions = []
            while input_video.isOpened(): 
                ret, frame = input_video.read() 
                if not ret: 
                    break

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


                # Pad with -69 if hands are not detected
                padding = [0] * (21 * 3)  # 21 landmarks * 3 coordinates (x, y, z)
                if not left_hand_data:
                    left_hand_data = padding
                if not right_hand_data:
                    right_hand_data = padding

                # Combine the data into a single row (Left hand first, then Right hand)
                row_data.extend(left_hand_data)
                row_data.extend(right_hand_data)

                writer.writerow(row_data) 
                frame_number += 1


            
                    
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Extract hand landmarks from video.')
    # parser.add_argument('video_path', type=str, help='Path to input video file.')
    # parser.add_argument('output_csv', type=str, help='Path to output CSV file.')
    # args = parser.parse_args()
    input_videos = [
        "input_videos/howareu.mp4", 
        "input_videos/nicetomeetu.mp4", 
        "input_videos/mynameis.mp4"
    ]
    csv_files = [
    "output_csv/howareyou.csv", 
    "output_csv/nicetomeetu.csv", 
    "output_csv/mynameis.csv"
    ]
    for i in range(3): 
        process_video(input_videos[i], csv_files[i])
