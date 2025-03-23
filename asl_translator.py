#!/usr/bin/env python3
"""
ASL Translator - Real-time American Sign Language Translation
This script uses computer vision and deep learning to recognize and translate ASL alphabet 
letters and common signs into text and speech.
"""

import cv2
import numpy as np
import time
from collections import deque
import os
import argparse
import pyttsx3

# Constants
LETTER_THRESHOLD = 3  # Frames required for stable letter detection

# Configure argument parsing
parser = argparse.ArgumentParser(description='ASL Translator')
parser.add_argument('--webcam', type=int, default=0, help='Webcam index to use (default: 0)')
parser.add_argument('--alphabet-model', type=str, default='models/asl_alphabet_CNN.h5', help='Path to alphabet model')
parser.add_argument('--no-tts', action='store_true', help='Disable text-to-speech')
args = parser.parse_args()

def load_alphabet_model(model_path):
    """
    Load the pre-trained ASL alphabet model.
    Args:
        model_path: Path to the model file
    Returns:
        The loaded model
    """
    print("Loading alphabet model...")
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"Warning: Alphabet model not found at {model_path}")
        print("Download it from: https://github.com/MelihGulum/ASL-Recognition-CNN-OpenCV")
        return None
    
    try:
        # Attempt to import TensorFlow and load the model
        try:
            import tensorflow as tf
            # Try to use MPS if available
            try:
                physical_devices = tf.config.list_physical_devices('GPU')
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                print("Using CPU for model inference")
            
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            print(f"Loaded alphabet model from {model_path}")
            return model
        except ImportError:
            # If TF is not available, try to load the model in OpenCV DNN format
            print("TensorFlow not available. Attempting alternative loading method...")
            model = cv2.dnn.readNetFromTensorflow(model_path)
            print(f"Loaded alphabet model with OpenCV DNN from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    except Exception as e:
        print(f"Could not load model: {e}")
        return None

def initialize_alphabet_labels():
    """
    Initialize the mapping of model indices to ASL letters.
    Returns:
        list: alphabet labels
    """
    # Alphabet (A-Z)
    alphabet_labels = [chr(ord('A') + i) for i in range(26)]
    # Add special tokens (typically space, delete, nothing)
    alphabet_labels.extend(["SPACE", "DELETE", "NOTHING"])
    
    return alphabet_labels

def setup_tts():
    """
    Set up the text-to-speech engine.
    Returns:
        pyttsx3.Engine: TTS engine instance
    """
    if args.no_tts:
        return None
    
    try:
        engine = pyttsx3.init()
        return engine
    except Exception as e:
        print(f"Warning: Could not initialize text-to-speech: {e}")
        return None

def speak(engine, text):
    """
    Speak the given text using the TTS engine.
    Args:
        engine: TTS engine instance
        text: Text to speak
    """
    if engine and not args.no_tts:
        engine.say(text)
        engine.runAndWait()

def preprocess_frame_for_alphabet(frame, model):
    """
    Preprocess a frame for the alphabet model.
    Args:
        frame: Input frame
        model: The loaded model (to determine right preprocessing)
    Returns:
        numpy.ndarray: Preprocessed frame ready for model input
    """
    # Check if we're using TensorFlow or OpenCV DNN
    if 'tensorflow' in str(type(model)):
        # For TensorFlow model
        # Resize frame for alphabet model (typically 64x64 or 200x200)
        small_frame = cv2.resize(frame, (64, 64))
        # Convert BGR (OpenCV) to RGB (model) and normalize [0,1]
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB) / 255.0
        frame_rgb = np.expand_dims(frame_rgb, axis=0)  # shape (1, 64, 64, 3)
        return frame_rgb
    else:
        # For OpenCV DNN model
        blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (64, 64), swapRB=True)
        return blob

def predict_letter(model, preprocessed_frame, alphabet_labels):
    """
    Predict the letter from the preprocessed frame.
    Args:
        model: Loaded model
        preprocessed_frame: Preprocessed input frame
        alphabet_labels: List of alphabet labels
    Returns:
        tuple: (letter, confidence)
    """
    if model is None:
        return None, 0.0
    
    try:
        if 'tensorflow' in str(type(model)):
            # TensorFlow model prediction
            letter_probs = model.predict(preprocessed_frame, verbose=0)  # shape (1, 29)
            letter_idx = int(np.argmax(letter_probs[0]))
            letter_conf = float(letter_probs[0][letter_idx])
        else:
            # OpenCV DNN model prediction
            model.setInput(preprocessed_frame)
            letter_probs = model.forward()
            letter_idx = int(np.argmax(letter_probs[0]))
            letter_conf = float(letter_probs[0][letter_idx])
        
        # Map index to letter (assuming 0->A, 1->B, ..., 25->Z)
        if letter_idx < len(alphabet_labels):
            letter = alphabet_labels[letter_idx]
        else:
            letter = f"Unknown_{letter_idx}"
        
        return letter, letter_conf
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, 0.0

def draw_info(frame, letter=None, spelled_word=None, sign=None, fps=None):
    """
    Draw information on the frame for display.
    Args:
        frame: Input frame
        letter: Current letter
        spelled_word: Current spelled word
        sign: Current recognized sign
        fps: FPS information
    Returns:
        numpy.ndarray: Frame with information drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    
    # Draw background rectangle for text
    cv2.rectangle(frame, (0, h-150), (w, h), (0, 0, 0), -1)
    
    # Draw FPS
    if fps:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h-120), font, 0.7, (255, 255, 255), 2)
    
    # Draw current letter
    if letter:
        cv2.putText(frame, f"Letter: {letter}", (10, h-90), font, 0.7, (255, 255, 255), 2)
    
    # Draw spelled word
    if spelled_word:
        cv2.putText(frame, f"Spelled: {spelled_word}", (10, h-60), font, 0.7, (255, 255, 255), 2)
    
    # Draw recognized sign
    if sign:
        cv2.putText(frame, f"Sign: {sign.upper()}", (10, h-30), font, 0.7, (0, 255, 0), 2)
    
    return frame

def draw_hand_roi(frame, roi=None):
    """
    Draw the hand region of interest on the frame.
    Args:
        frame: Input frame
        roi: Region of interest coordinates (x, y, w, h)
    Returns:
        numpy.ndarray: Frame with ROI drawn
    """
    if roi:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw a reference rectangle in the center of the frame
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    roi_size = min(w, h) // 3
    cv2.rectangle(frame, 
                 (center_x - roi_size, center_y - roi_size),
                 (center_x + roi_size, center_y + roi_size),
                 (255, 0, 0), 2)
    
    return frame

def main():
    """
    Main function for ASL translator.
    """
    # Load alphabet model
    alphabet_model = load_alphabet_model(args.alphabet_model)
    
    # Initialize alphabet labels
    alphabet_labels = initialize_alphabet_labels()
    
    # Setup TTS engine
    tts_engine = setup_tts()
    
    # Initialize video capture from webcam
    print(f"Opening webcam {args.webcam}...")
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {args.webcam}")
        return
    
    # Set a higher resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variables for smoothing outputs
    last_letter = None
    letter_count = 0
    spelled_word = ""  # to accumulate letters into a word when finger-spelling
    
    # For FPS calculation
    prev_time = time.time()
    frame_count = 0
    fps = 0
    
    print("Starting ASL translation. Press 'q' to quit, 'c' to clear spelled word.")
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(args.webcam)
                if not cap.isOpened():
                    print(f"Error: Could not reconnect to webcam {args.webcam}")
                    break
                continue
            
            # Flip the frame for mirror effect (more intuitive for user)
            frame = cv2.flip(frame, 1)
            
            # Extract hand region (center of the frame)
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            roi_size = min(w, h) // 3
            hand_roi = (center_x - roi_size, center_y - roi_size, roi_size * 2, roi_size * 2)
            hand_frame = frame[hand_roi[1]:hand_roi[1]+hand_roi[3], hand_roi[0]:hand_roi[0]+hand_roi[2]]
            
            # Process for alphabet recognition
            letter = None
            if alphabet_model is not None and hand_frame.size > 0:
                preprocessed_frame = preprocess_frame_for_alphabet(hand_frame, alphabet_model)
                letter, letter_conf = predict_letter(alphabet_model, preprocessed_frame, alphabet_labels)
                
                # Process the detected letter
                if letter and letter not in ["NOTHING"]:
                    if letter == last_letter:
                        letter_count += 1
                    else:
                        last_letter = letter
                        letter_count = 1
                    
                    # Only add letter to spelled_word if it's been stable for a few frames
                    if letter_count >= LETTER_THRESHOLD:
                        if letter == "SPACE":
                            # Space detected, finalize the spelled word
                            if spelled_word:
                                print(f"Recognized spelled word: {spelled_word}")
                                speak(tts_engine, spelled_word)
                                spelled_word = ""
                        elif letter == "DELETE" and spelled_word:
                            # Delete the last character
                            spelled_word = spelled_word[:-1]
                            print(f"Deleted last letter. Spelled so far: {spelled_word}")
                        elif letter not in ["SPACE", "DELETE"]:
                            # append letter to the spelled word
                            spelled_word += letter
                            print(f"Spelled so far: {spelled_word}")
                        
                        # Reset count to avoid multiple appends
                        letter_count = 0
            
            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - prev_time
            if elapsed >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = current_time
            
            # Draw ROI on frame
            frame = draw_hand_roi(frame, hand_roi)
            
            # Draw information on frame
            frame = draw_info(frame, letter, spelled_word, None, fps)
            
            # Draw instructions
            cv2.putText(frame, "Position hand in blue box", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 'c' to clear", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('ASL Translator', frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):  # Clear spelled word
                spelled_word = ""
                print("Cleared spelled word")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        print("ASL translator stopped")

if __name__ == "__main__":
    main()
