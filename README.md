# Real-Time ASL Gesture to Text Translation System

![original](https://github.com/user-attachments/assets/b53a487b-70b4-4e27-a269-b8b055cb8f37)

![original](https://github.com/user-attachments/assets/d641b5fd-18ee-4c66-858b-bd24eb6daca1)

## Project Overview

This project is a real-time American Sign Language (ASL) gesture translation system that converts hand gestures captured through a webcam into text and spoken language. The system uses computer vision and deep learning techniques to recognize both static signs (individual letters and numbers) and dynamic gestures (words or phrases that involve movement over time).

## Model Architecture

The system employs a multi-model approach to achieve high accuracy in ASL translation:

1. **Landmark-based Model**: A neural network that processes hand landmark coordinates extracted by MediaPipe. It consists of:
   - Dense layers with batch normalization and dropout for regularization
   - Input features representing 21 landmarks × 3 coordinates (x, y, z) × 2 hands
   - Optimized for static gesture recognition

2. **Sequence Model**: An LSTM-based neural network for dynamic gesture recognition:
   - Processes sequences of hand landmarks (30 frames, approximately 1 second)
   - Captures temporal patterns in gestures that involve movement
   - Two LSTM layers followed by dense layers

3. **Image Model** (Optional): A MobileNetV2-based CNN for processing hand images:
   - Transfer learning from pre-trained ImageNet weights
   - Fine-tuned for ASL gesture recognition
   - Useful for challenging lighting conditions or when landmark detection is difficult

The system uses a voting mechanism to combine predictions from these models, improving overall accuracy and robustness.

## Key Features

- **Real-time translation**: Processes webcam feed at interactive framerates
- **Multi-hand support**: Can detect and process two hands simultaneously
- **Interactive training mode**: Add new signs and customize the model for personal use
- **Text-to-speech output**: Spoken output of recognized signs with configurable cooldown
- **Dynamic gesture recognition**: Captures signs that involve movement over time
- **Modern GUI**: User-friendly interface with visualization of hand landmarks
- **Cross-platform compatibility**: Works on Windows, macOS, and Linux

## Technical Implementation

- **Hand tracking**: Uses MediaPipe Hands for accurate hand landmark detection
- **Data preprocessing**: Normalizes hand landmarks and extracts regions of interest
- **Model training**: Supports both loading pre-trained models and on-the-fly training
- **Inference optimization**: Uses efficient prediction pipelines for real-time performance

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/my-augmented-voice.git
   cd my-augmented-voice
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Note: The requirements file automatically handles platform-specific dependencies, including TensorFlow for Apple Silicon.

3. Create necessary directories (first run will do this automatically):

   ```bash
   mkdir -p asl_translator/model
   ```

## Usage

### Basic Recognition Mode

Run the application in standard recognition mode:

```bash
python asl_cnn_translator.py
```

This will launch the GUI application ready to recognize ASL signs.

### Interactive Training Mode

To add new signs or improve existing ones:

```bash
python asl_cnn_translator.py --interactive
```

In this mode, you can:

- Create new sign labels
- Collect training examples by demonstrating the sign
- Train the models with your collected data
- Test recognition immediately

### Retrain Models

To create fresh models instead of loading existing ones:

```bash
python asl_cnn_translator.py --retrain
```

This is useful if you want to start with a clean slate or if the existing models have become corrupted.

## Interface Controls

- **Toggle Camera**: Start/stop the webcam feed
- **Train Model**: In interactive mode, train the model with collected examples
- **Collect Examples**: Capture frames for a specific sign in interactive mode
- **Delete Sign**: Remove a sign from the dataset
- **Toggle TTS**: Enable/disable text-to-speech output
- **Toggle Dynamic Recognition**: Switch between static and dynamic gesture recognition

## Project Structure

- `asl_cnn_translator.py`: Main application file containing model definitions and GUI
- `requirements.txt`: Dependencies required to run the application
- `asl_translator/model/`: Directory for storing trained models and training data
  - `dual_hand_model.keras`: Landmark-based model for static gesture recognition
  - `dynamic_gesture_model.keras`: LSTM model for dynamic gesture recognition
  - `image_model.keras`: CNN model for image-based recognition
  - `labels.pkl`: Mapping between class indices and sign labels
  - `training_data/`: Contains collected training examples
