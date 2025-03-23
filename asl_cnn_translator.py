import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM, Input, Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pickle
import time
import pyttsx3
import random
import argparse
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import glob

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_SIZE = 224
NUM_LANDMARKS = 21
MODEL_PATH = "asl_translator/model/"
LANDMARK_MODEL_PATH = os.path.join(MODEL_PATH, "dual_hand_model.keras")
SEQUENCE_MODEL_PATH = os.path.join(MODEL_PATH, "dynamic_gesture_model.keras")
IMAGE_MODEL_PATH = os.path.join(MODEL_PATH, "image_model.keras")
LABELS_PATH = os.path.join(MODEL_PATH, "labels.pkl")
TRAIN_DATA_PATH = os.path.join(MODEL_PATH, "training_data")
SEQUENCE_LENGTH = 30  # For dynamic gestures - 30 frames = ~1 second

class ASLCNNTranslator:
    def __init__(self, load_existing=True, interactive_mode=False):
        self.interactive_mode = interactive_mode
        self.load_existing = load_existing
        
        self.current_training_example = None
        self.current_training_label = None
        self.is_collecting = False
        self.collected_frames = []
        
        self.training_in_progress = False
        
        self.last_sign = None
        self.sign_count = 0
        self.min_detection_count = 3
        self.last_spoken_time = 0
        self.cooldown = 2  # seconds
        
        self.landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.use_dynamic_recognition = True
        
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
        
        self._setup_models()
        
    def _setup_models(self):
        if self.load_existing and os.path.exists(LABELS_PATH):
            try:
                with open(LABELS_PATH, 'rb') as f:
                    self.labels = pickle.load(f)
                print(f"Loaded existing labels: {self.labels}")
            except Exception as e:
                print(f"Error loading labels: {e}")
                self.labels = []
        else:
            self.labels = []
            
        self.num_classes = max(1, len(self.labels))  # Ensure at least 1 class
        
        # Create mappings
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
        
        # Create or load landmark model
        if self.load_existing and os.path.exists(LANDMARK_MODEL_PATH):
            try:
                print(f"Loading existing landmark model from {LANDMARK_MODEL_PATH}")
                self.landmark_model = tf.keras.models.load_model(LANDMARK_MODEL_PATH)
            except Exception as e:
                print(f"Error loading landmark model: {e}")
                print("Creating new landmark model...")
                self.landmark_model = self._create_landmark_model()
        else:
            print("Creating new landmark model...")
            self.landmark_model = self._create_landmark_model()
            
        # Create or load sequence model for dynamic gestures
        if self.load_existing and os.path.exists(SEQUENCE_MODEL_PATH):
            try:
                print(f"Loading existing sequence model from {SEQUENCE_MODEL_PATH}")
                self.sequence_model = tf.keras.models.load_model(SEQUENCE_MODEL_PATH)
            except Exception as e:
                print(f"Error loading sequence model: {e}")
                print("Creating new sequence model...")
                self.sequence_model = self._create_sequence_model()
        else:
            print("Creating new sequence model...")
            self.sequence_model = self._create_sequence_model()
        
    def _create_landmark_model(self):
        """Create a model for landmark-based recognition"""
        # Model for hand landmarks (x, y, z for each of the 21 landmarks = 63 features)
        # We'll use two hands, so we'll double the input size
        input_dim = NUM_LANDMARKS * 3 * 2  # 21 landmarks * 3 coordinates * 2 hands
        
        # Make sure we have at least 1 class to predict
        num_output_classes = max(1, self.num_classes)
        print(f"Creating landmark model with {num_output_classes} output classes")
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_output_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_image_model(self):
        base_model = MobileNetV2(
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_sequence_model(self):
        input_dim = NUM_LANDMARKS * 3 * 2  # 21 landmarks * 3 coordinates * 2 hands
        num_output_classes = max(1, self.num_classes)
        print(f"Creating sequence model with {num_output_classes} output classes")
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, input_dim)),
            Dropout(0.3),
            LSTM(32),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(num_output_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _preprocess_landmarks(self, hand_landmarks_list):
        if not hand_landmarks_list:
            return None
            
        feature_vector = np.zeros(NUM_LANDMARKS * 3 * 2)
        
        for i, hand_landmarks in enumerate(hand_landmarks_list[:2]):
            for j, landmark in enumerate(hand_landmarks.landmark):
                idx = i * NUM_LANDMARKS * 3 + j * 3
                feature_vector[idx] = landmark.x
                feature_vector[idx + 1] = landmark.y
                feature_vector[idx + 2] = landmark.z
                
        return feature_vector
    
    def _preprocess_hand_image(self, frame, hand_landmarks):
        h, w, _ = frame.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        if x_max <= x_min or y_max <= y_min:
            return None
            
        hand_img = frame[y_min:y_max, x_min:x_max]
        hand_img = cv2.resize(hand_img, (IMAGE_SIZE, IMAGE_SIZE))
        
        if len(hand_img.shape) == 3 and hand_img.shape[2] == 3:
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        
        hand_img = hand_img / 255.0
        
        return hand_img
    
    def recognize_sign(self, frame, hand_landmarks_list):
        if len(self.labels) == 0:
            return "No signs trained yet", 0.0
        
        landmark_features = self._preprocess_landmarks(hand_landmarks_list)
        
        if landmark_features is None or landmark_features.size == 0:
            return None, 0.0
            
        self.landmark_buffer.append(landmark_features)
        
        landmark_features_reshaped = landmark_features.reshape(1, -1)
        
        try:
            prediction = self.landmark_model.predict(landmark_features_reshaped, verbose=0)
            predicted_idx = np.argmax(prediction[0])
            confidence = prediction[0][predicted_idx]
            
            if (self.use_dynamic_recognition and 
                len(self.landmark_buffer) == SEQUENCE_LENGTH and 
                self.sequence_model is not None):
                sequence = np.array(list(self.landmark_buffer))
                sequence = sequence.reshape(1, SEQUENCE_LENGTH, -1)
                
                try:
                    sequence_prediction = self.sequence_model.predict(sequence, verbose=0)
                    sequence_predicted_idx = np.argmax(sequence_prediction[0])
                    sequence_confidence = sequence_prediction[0][sequence_predicted_idx]
                    
                    if sequence_confidence > confidence:
                        predicted_idx = sequence_predicted_idx
                        confidence = sequence_confidence
                except Exception as e:
                    print(f"Error in sequence prediction: {e}")
            
            if confidence > 0.7:
                return self.idx_to_label.get(predicted_idx, "Unknown"), confidence
            return None, 0.0
        except Exception as e:
            print(f"Error in sign recognition: {e}")
            return None, 0.0
    
    def start_collecting(self, label):
        self.is_collecting = True
        self.current_training_label = label
        self.collected_frames = []
        self.landmark_buffer.clear()
        print(f"Started collecting examples for: {label}")
    
    def stop_collecting(self):
        self.is_collecting = False
        print(f"Stopped collecting. Got {len(self.collected_frames)} frames for {self.current_training_label}")
        
        if len(self.collected_frames) > 0:
            self._save_training_data()
            
        self.collected_frames = []
        self.current_training_label = None
    
    def _save_training_data(self):
        """Save collected training data"""
        label_dir = os.path.join(TRAIN_DATA_PATH, self.current_training_label)
        os.makedirs(label_dir, exist_ok=True)
        
        print(f"Saving {len(self.collected_frames)} frames to {label_dir}")
        
        saved_count = 0
        for i, (landmark_data, _) in enumerate(self.collected_frames):
            if landmark_data is None or landmark_data.size == 0:
                print(f"Skipping empty frame #{i}")
                continue
                
            timestamp = int(time.time() * 1000)
            random_suffix = random.randint(1000, 9999)
            filename = f"{self.current_training_label}_{timestamp}_{random_suffix}_{i}.npy"
            filepath = os.path.join(label_dir, filename)
            
            try:
                np.save(filepath, landmark_data)
                saved_count += 1
            except Exception as e:
                print(f"Error saving frame {i}: {e}")
                
        print(f"Saved {saved_count} examples for {self.current_training_label}")
        
        if self.current_training_label not in self.labels:
            print(f"Adding new label: {self.current_training_label}")
            self.labels.append(self.current_training_label)
            self.num_classes = len(self.labels)
            self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
            self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
            
            try:
                with open(LABELS_PATH, 'wb') as f:
                    pickle.dump(self.labels, f)
            except Exception as e:
                print(f"Error saving labels: {e}")
    
    def train_model(self):
        """
        Train or retrain both static and dynamic models on collected data
        """
        if os.path.exists(TRAIN_DATA_PATH):
            print("Training models on collected data...")
            self.training_in_progress = True
            
            X_train = []
            y_train = []
            sequences = []
            sequence_labels = []
            
            unique_labels = set()
            for label in os.listdir(TRAIN_DATA_PATH):
                label_dir = os.path.join(TRAIN_DATA_PATH, label)
                if os.path.isdir(label_dir):
                    unique_labels.add(label)
            
            print(f"Found {len(unique_labels)} unique labels: {sorted(list(unique_labels))}")
            for label in unique_labels:
                if label not in self.labels:
                    print(f"Found new label: {label}")
                    self.labels.append(label)
            
            self.num_classes = len(self.labels)
            self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
            self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
            
            if self.num_classes > 0:
                print(f"Rebuilding models for {self.num_classes} classes...")
                self.landmark_model = self._create_landmark_model()
                self.sequence_model = self._create_sequence_model()
            
            for label in os.listdir(TRAIN_DATA_PATH):
                label_dir = os.path.join(TRAIN_DATA_PATH, label)
                if not os.path.isdir(label_dir):
                    continue
                    
                frame_files = glob.glob(os.path.join(label_dir, "*.npy"))
                if not frame_files:
                    continue
                    
                label_idx = self.label_to_idx.get(label, 0)
                
                curr_sequence = []
                
                for frame_file in frame_files:
                    try:
                        landmarks = np.load(frame_file)
                        
                        X_train.append(landmarks)
                        y_train.append(label_idx)
                        
                        curr_sequence.append(landmarks)
                        
                        if len(curr_sequence) >= SEQUENCE_LENGTH:
                            sequences.append(curr_sequence[-SEQUENCE_LENGTH:])
                            sequence_labels.append(label_idx)
                            
                    except Exception as e:
                        print(f"Error loading {frame_file}: {e}")
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            print(f"Collected {len(X_train)} static examples, {len(sequences)} sequences")
            print(f"Number of classes: {self.num_classes}")
            
            if len(X_train) > 0:
                y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
                
                callbacks = [
                    ModelCheckpoint(
                        filepath=LANDMARK_MODEL_PATH, 
                        save_best_only=True, 
                        monitor='val_accuracy'
                    )
                ]
                
                indices = np.arange(len(X_train))
                np.random.shuffle(indices)
                split = int(0.8 * len(indices))
                train_indices = indices[:split]
                val_indices = indices[split:]
                
                X_train_split = X_train[train_indices]
                y_train_split = y_train_one_hot[train_indices]
                X_val = X_train[val_indices] if len(val_indices) > 0 else X_train_split
                y_val = y_train_one_hot[val_indices] if len(val_indices) > 0 else y_train_split
                
                print("Training static landmark model...")
                history = self.landmark_model.fit(
                    X_train_split, y_train_split,
                    epochs=15,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks
                )
                
                try:
                    self.landmark_model.save(LANDMARK_MODEL_PATH)
                    print(f"Static model trained and saved to {LANDMARK_MODEL_PATH}")
                except Exception as e:
                    print(f"Error saving static model: {e}")
                
                if len(sequences) > 0:
                    print(f"Training dynamic gesture model with {len(sequences)} sequences...")
                    sequences = np.array(sequences)
                    sequence_labels = np.array(sequence_labels)
                    sequence_labels_one_hot = tf.keras.utils.to_categorical(sequence_labels, num_classes=self.num_classes)
                    
                    seq_indices = np.arange(len(sequences))
                    np.random.shuffle(seq_indices)
                    seq_split = int(0.8 * len(seq_indices))
                    seq_train_indices = seq_indices[:seq_split]
                    seq_val_indices = seq_indices[seq_split:]
                    
                    X_seq_train = sequences[seq_train_indices]
                    y_seq_train = sequence_labels_one_hot[seq_train_indices]
                    X_seq_val = sequences[seq_val_indices] if len(seq_val_indices) > 0 else X_seq_train
                    y_seq_val = sequence_labels_one_hot[seq_val_indices] if len(seq_val_indices) > 0 else y_seq_train
                    
                    seq_callbacks = [
                        ModelCheckpoint(
                            filepath=SEQUENCE_MODEL_PATH, 
                            save_best_only=True, 
                            monitor='val_accuracy'
                        )
                    ]
                    
                    self.sequence_model.fit(
                        X_seq_train, y_seq_train,
                        epochs=15,
                        batch_size=16,
                        validation_data=(X_seq_val, y_seq_val),
                        callbacks=seq_callbacks
                    )
                    
                    try:
                        self.sequence_model.save(SEQUENCE_MODEL_PATH)
                        print(f"Dynamic gesture model trained and saved to {SEQUENCE_MODEL_PATH}")
                    except Exception as e:
                        print(f"Error saving dynamic model: {e}")
                else:
                    print("Not enough sequence data to train dynamic gesture model")
            else:
                print("No training data found.")
            
            self.training_in_progress = False
            return True
        else:
            print("No training data directory found.")
            self.training_in_progress = False
            return False
    
    def collect_example(self, frame, hand_landmarks_list):
        if self.is_collecting and hand_landmarks_list:
            print(f"Collecting frame with {len(hand_landmarks_list)} hands detected")
            
            landmark_features = self._preprocess_landmarks(hand_landmarks_list)
            
            if landmark_features is not None and landmark_features.size > 0:
                self.collected_frames.append((landmark_features, None))
                print(f"Successfully collected frame #{len(self.collected_frames)} for {self.current_training_label}")
                return True
            else:
                print("Failed to extract landmarks from current frame")
        return False
                
    def process_frame(self, frame, results, speak_function):
        display_frame = frame.copy()
        recognized_sign = None
        confidence = 0
        
        mode_text = "Interactive Training" if self.interactive_mode else "Recognition"
        status_text = f"Mode: {mode_text} | Dynamic Gestures: {'ON' if self.use_dynamic_recognition else 'OFF'}"
        status_color = (0, 255, 0)  # Green
        cv2.putText(display_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if self.interactive_mode:
            if self.is_collecting:
                training_text = f"Collecting: {self.current_training_label} ({len(self.collected_frames)} frames)"
                cv2.putText(display_frame, training_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif self.training_in_progress:
                cv2.putText(display_frame, "Training in progress...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            else:
                cv2.putText(display_frame, "Press 'c' to start collecting, 't' to train", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.use_dynamic_recognition:
            buffer_status = f"Sequence Buffer: {len(self.landmark_buffer)}/{SEQUENCE_LENGTH} frames"
            cv2.putText(display_frame, buffer_status, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        hands_text = f"Hands Detected: {num_hands}"
        cv2.putText(display_frame, hands_text, (frame.shape[1] - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
        if results.multi_hand_landmarks:
            hand_landmarks_list = results.multi_hand_landmarks
            
            for i, hand_landmarks in enumerate(hand_landmarks_list):
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                h, w, _ = frame.shape
                hand_label = f"Hand {i+1}"
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(display_frame, hand_label, (wrist_x, wrist_y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if self.is_collecting:
                success = self.collect_example(frame, hand_landmarks_list)
                if success:
                    cv2.putText(display_frame, "Frame Collected!", (frame.shape[1] - 250, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                recognized_sign, confidence = self.recognize_sign(frame, hand_landmarks_list)
        
        if recognized_sign:
            if recognized_sign == self.last_sign:
                self.sign_count += 1
            else:
                self.last_sign = recognized_sign
                self.sign_count = 1
                
            if self.sign_count >= self.min_detection_count:
                current_time = time.time()
                if current_time - self.last_spoken_time >= self.cooldown:
                    if not self.is_collecting:
                        self.last_spoken_time = current_time
                        speak_function(recognized_sign)
                    
                cv2.rectangle(display_frame, (10, 120), (400, 170), (0, 0, 0), -1)
                cv2.putText(display_frame, f"{recognized_sign} ({confidence:.2f})", (20, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return display_frame

    def delete_sign_data(self, sign_name=None):
        if sign_name is None or sign_name not in self.labels:
            print(f"Sign '{sign_name}' not found or no sign specified")
            return False
            
        sign_dir = os.path.join(TRAIN_DATA_PATH, sign_name)
        if os.path.exists(sign_dir) and os.path.isdir(sign_dir):
            try:
                for file in os.listdir(sign_dir):
                    os.remove(os.path.join(sign_dir, file))
                os.rmdir(sign_dir)
                
                self.labels.remove(sign_name)
                
                self.num_classes = len(self.labels)
                self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
                self.idx_to_label = {i: label for i, label in enumerate(self.labels)}
                
                with open(LABELS_PATH, 'wb') as f:
                    pickle.dump(self.labels, f)
                    
                print(f"Successfully deleted data for sign: {sign_name}")
                return True
            except Exception as e:
                print(f"Error deleting sign data: {e}")
                return False
        else:
            print(f"Training data directory for '{sign_name}' not found")
            return False

def setup_camera():
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def setup_tts():
    engine = pyttsx3.init()
    
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    
    def speak(text):
        print(f"Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
    
    return speak

class ASLTranslatorGUI:
    def __init__(self, root, interactive=False, retrain=False):
        self.root = root
        self.root.title("ASL CNN Translator")
        self.root.geometry("1024x768")
        self.root.configure(bg="#f0f0f0")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', 
                             background='#4a6ea9', 
                             foreground='white', 
                             font=('Arial', 10, 'bold'),
                             padding=5,
                             borderwidth=0)
        self.style.map('TButton', 
                      background=[('active', '#5a7eb9'), ('pressed', '#3a5e99')])
        self.style.configure('Status.TLabel', 
                            background='#f0f0f0', 
                            foreground='#333333', 
                            font=('Arial', 10))
        self.style.configure('Header.TLabel', 
                            background='#f0f0f0', 
                            foreground='#333333', 
                            font=('Arial', 16, 'bold'))
        self.style.configure('Info.TLabel', 
                            background='#f0f0f0', 
                            foreground='#555555', 
                            font=('Arial', 9))
        
        self.interactive = interactive
        self.running = False
        self.cap = None
        self.current_frame = None
        self.current_label = None
        self.fps = 0
        self.prev_time = time.time()
        self.frame_count = 0
        
        self.create_widgets()
        
        self.translator = ASLCNNTranslator(load_existing=not retrain, interactive_mode=interactive)
        
        self.tts_engine = setup_tts()
        
        self.hands = mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2)
            
        if self.interactive:
            self.update_status("Interactive mode enabled. Ready to collect training data.")
        else:
            self.update_status("Recognition mode active. Ready to detect ASL signs.")
            
        if not self.interactive:
            for widget in self.training_frame.winfo_children():
                widget.configure(state="disabled")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.Frame(main_frame, style='TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_header = ttk.Label(left_frame, text="ASL Recognition", style='Header.TLabel')
        video_header.pack(pady=(0, 10), anchor=tk.W)
        
        self.canvas = tk.Canvas(left_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        recognition_frame = ttk.Frame(left_frame, style='TFrame', padding=5)
        recognition_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(recognition_frame, text="Detected Sign:", style='Status.TLabel').grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.detected_sign_var = tk.StringVar(value="None")
        ttk.Label(recognition_frame, textvariable=self.detected_sign_var, style='Status.TLabel').grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(recognition_frame, text="Confidence:", style='Status.TLabel').grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.confidence_var = tk.StringVar(value="0%")
        ttk.Label(recognition_frame, textvariable=self.confidence_var, style='Status.TLabel').grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(recognition_frame, text="Hands Detected:", style='Status.TLabel').grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.hands_detected_var = tk.StringVar(value="0")
        ttk.Label(recognition_frame, textvariable=self.hands_detected_var, style='Status.TLabel').grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(recognition_frame, text="FPS:", style='Status.TLabel').grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.fps_var = tk.StringVar(value="0")
        ttk.Label(recognition_frame, textvariable=self.fps_var, style='Status.TLabel').grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        right_frame = ttk.Frame(main_frame, width=300, style='TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        right_frame.pack_propagate(False)
        
        controls_header = ttk.Label(right_frame, text="Controls", style='Header.TLabel')
        controls_header.pack(pady=(0, 10), anchor=tk.W)
        
        controls_frame = ttk.Frame(right_frame, style='TFrame', padding=5)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.start_stop_button = ttk.Button(controls_frame, text="Start", command=self.toggle_camera)
        self.start_stop_button.pack(fill=tk.X, pady=5)
        
        self.dynamic_rec_var = tk.BooleanVar(value=True)
        self.dynamic_button = ttk.Checkbutton(
            controls_frame, 
            text="Dynamic Gesture Recognition", 
            variable=self.dynamic_rec_var,
            command=self.toggle_dynamic_recognition)
        self.dynamic_button.pack(fill=tk.X, pady=5)
        
        self.speak_button = ttk.Button(
            controls_frame, 
            text="Speak Last Prediction", 
            command=self.speak_last_prediction)
        self.speak_button.pack(fill=tk.X, pady=5)
        
        training_header = ttk.Label(right_frame, text="Training", style='Header.TLabel')
        training_header.pack(pady=(20, 10), anchor=tk.W)
        
        self.training_frame = ttk.Frame(right_frame, style='TFrame', padding=5)
        self.training_frame.pack(fill=tk.X, pady=5)
        
        self.collect_button = ttk.Button(
            self.training_frame, 
            text="Collect Examples", 
            command=self.collect_examples)
        self.collect_button.pack(fill=tk.X, pady=5)
        
        self.stop_collect_button = ttk.Button(
            self.training_frame, 
            text="Stop Collecting", 
            command=self.stop_collecting,
            state="disabled")
        self.stop_collect_button.pack(fill=tk.X, pady=5)
        
        self.train_button = ttk.Button(
            self.training_frame, 
            text="Train Model", 
            command=self.train_model)
        self.train_button.pack(fill=tk.X, pady=5)
        
        self.delete_sign_button = ttk.Button(
            self.training_frame, 
            text="Delete Sign", 
            command=self.delete_sign)
        self.delete_sign_button.pack(fill=tk.X, pady=5)
        
        status_frame = ttk.Frame(right_frame, style='TFrame')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        status_header = ttk.Label(status_frame, text="Status", style='Header.TLabel')
        status_header.pack(pady=(0, 5), anchor=tk.W)
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                     style='Status.TLabel', wraplength=280)
        self.status_label.pack(fill=tk.X, pady=5)
        
        self.collection_info_var = tk.StringVar(value="")
        self.collection_info = ttk.Label(status_frame, textvariable=self.collection_info_var, 
                                        style='Info.TLabel', wraplength=280)
        self.collection_info.pack(fill=tk.X, pady=5)
        
        self.sequence_info_var = tk.StringVar(value="Sequence Buffer: 0/30")
        self.sequence_info = ttk.Label(status_frame, textvariable=self.sequence_info_var, 
                                      style='Info.TLabel')
        self.sequence_info.pack(fill=tk.X, pady=5)
        
    def update_status(self, message):
        self.status_var.set(message)
        
    def toggle_camera(self):
        if self.running:
            self.running = False
            self.start_stop_button.configure(text="Start")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.update_status("Camera stopped.")
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="Camera is OFF", 
                fill="white", 
                font=('Arial', 20))
        else:
            # Start camera
            self.cap = setup_camera()
            if self.cap is None:
                messagebox.showerror("Error", "Could not open camera.")
                return
                
            self.running = True
            self.start_stop_button.configure(text="Stop")
            self.update_status("Camera started.")
            self.process_video()
    
    def process_video(self):
        if not self.running:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.update_status("Error: Failed to capture image.")
            return
            
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(frame_rgb)
        
        processed_frame = self.translator.process_frame(frame, results, self.tts_engine)
        
        if self.translator.is_collecting:
            collected = len(self.translator.collected_frames)
            self.collection_info_var.set(f"Collecting for '{self.current_label}': {collected} frames")
        
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.prev_time >= 1:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = current_time
        
        self.fps_var.set(f"{self.fps}")
        
        hands_detected = 0 if results.multi_hand_landmarks is None else len(results.multi_hand_landmarks)
        self.hands_detected_var.set(str(hands_detected))
        
        if hasattr(self.translator, 'last_sign') and self.translator.last_sign:
            self.detected_sign_var.set(self.translator.last_sign)
            if hasattr(self.translator, 'last_confidence'):
                self.confidence_var.set(f"{self.translator.last_confidence:.1%}")
        
        buffer_size = len(self.translator.landmark_buffer)
        buffer_max = self.translator.landmark_buffer.maxlen
        self.sequence_info_var.set(f"Sequence Buffer: {buffer_size}/{buffer_max}")
        
        self.current_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width/2, canvas_height/2, image=self.photo, anchor=tk.CENTER)
        
        self.root.after(10, self.process_video)
        
    def collect_examples(self):
        if not self.running:
            messagebox.showwarning("Warning", "Start the camera first.")
            return
            
        label = simpledialog.askstring("Input", "Enter label for this sign:", parent=self.root)
        if not label:
            return
            
        label = label.strip().upper()
        self.current_label = label
        
        self.translator.collected_frames = []
        
        self.translator.start_collecting(label)
        self.collect_button.configure(state="disabled")
        self.stop_collect_button.configure(state="normal")
        self.update_status(f"Collecting examples for '{label}'...")
        
    def stop_collecting(self):
        self.translator.stop_collecting()
        self.collect_button.configure(state="normal")
        self.stop_collect_button.configure(state="disabled")
        
        collected = len(self.translator.collected_frames)
        self.collection_info_var.set(f"Collected {collected} frames for '{self.current_label}'")
        self.update_status("Stopped collecting examples.")
        
    def train_model(self):
        print(f"Collected frames: {len(self.translator.collected_frames)}")
        
        if not self.translator.collected_frames:
            if hasattr(self, 'hands_detected_var') and int(self.hands_detected_var.get()) > 0:
                self.update_status("No examples saved yet. Collecting a few examples now...")
                if not hasattr(self, 'current_label') or not self.current_label:
                    self.current_label = "DEFAULT"
                
                self.translator.start_collecting(self.current_label)
                self.root.after(2000, self.manual_collection_callback)
                return
            else:
                messagebox.showwarning("Warning", "No training data collected. Please collect examples first.")
                return
            
        self.train_button.configure(state="disabled")
        self.update_status("Training model... This may take a moment.")
        
        def train_thread():
            self.translator.train_model()
            self.root.after(0, lambda: self.train_button.configure(state="normal"))
            self.root.after(0, lambda: self.update_status("Training complete. Model updated."))
            
        threading.Thread(target=train_thread).start()
    
    def manual_collection_callback(self):
        self.translator.stop_collecting()
        collected = len(self.translator.collected_frames)
        self.collection_info_var.set(f"Auto-collected {collected} frames for '{self.current_label}'")
        
        if collected > 0:
            self.train_model()
        else:
            self.update_status("Could not collect any frames. Please ensure hands are visible.")
        
    def toggle_dynamic_recognition(self):
        self.translator.use_dynamic_recognition = self.dynamic_rec_var.get()
        status = "enabled" if self.translator.use_dynamic_recognition else "disabled"
        self.update_status(f"Dynamic gesture recognition {status}")
        self.translator.landmark_buffer.clear()
        
    def speak_last_prediction(self):
        if hasattr(self.translator, 'last_sign') and self.translator.last_sign:
            self.tts_engine(self.translator.last_sign)
        else:
            self.update_status("No sign detected yet.")
            
    def delete_sign(self):
        if not hasattr(self.translator, 'labels') or not self.translator.labels:
            self.update_status("No signs trained yet.")
            return
            
        sign_name = simpledialog.askstring(
            "Delete Sign", 
            f"Select a sign to delete:\n{', '.join(self.translator.labels)}",
            parent=self.root
        )
        
        if sign_name:
            if sign_name in self.translator.labels:
                success = self.translator.delete_sign_data(sign_name)
                if success:
                    self.update_status(f"Deleted training data for '{sign_name}'. Please retrain the model.")
                else:
                    self.update_status(f"Failed to delete '{sign_name}'.")
            else:
                self.update_status(f"Sign '{sign_name}' not found.")
    
    def on_closing(self):
        if self.running and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()
        self.root.destroy()

def main(interactive=False, retrain=False):
    print("Starting ASL CNN Translator...")
    
    root = tk.Tk()
    app = ASLTranslatorGUI(root, interactive=interactive, retrain=retrain)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASL CNN Translator')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive training mode')
    parser.add_argument('--retrain', action='store_true', help='Create new model instead of loading existing')
    args = parser.parse_args()
    
    main(interactive=args.interactive, retrain=args.retrain)
