import cv2
import csv
import argparse

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
    running_mode=VisionRunningMode.IMAGE)

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
                if results.hand_world_landmarks:  # Check if hand_world_landmarks exist
                    for hand_index, hand_world_landmarks in enumerate(results.hand_world_landmarks):
                        for landmark_index, landmark in enumerate(hand_world_landmarks):
                            hand_land_marked_positions.append((landmark.x, landmark.y, landmark.z))
                
                writer.writerow(hand_land_marked_positions)
                hand_land_marked_positions.clear()
                frame_number += 1

            
                    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract hand landmarks from video.')
    parser.add_argument('video_path', type=str, help='Path to input video file.')
    parser.add_argument('output_csv', type=str, help='Path to output CSV file.')
    args = parser.parse_args()
    
    process_video(args.video_path, args.output_csv)
