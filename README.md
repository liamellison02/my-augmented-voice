# Augmented Voice  
**Real-Time ASL Gesture to Text Translation System**  

---  

## Inspiration  
Communication is a fundamental human right, yet millions with hearing or speech impairments struggle in a world dominated by spoken language. ASL is more than a direct translation of English—it relies on complex gestures, facial expressions, and spatial positioning. Despite advances in computer vision, there is no consistent framework for accurately interpreting these multi-layered signs. Existing approaches focus on letter-based translations, making them impractical for dynamic conversations.  

We aimed to bridge this gap by building a system that empowers ASL users and can be expanded to support more natural, expressive communication.  

---  

## What It Does  

Our wearable camera captures and converts hand gestures into speech in real time, enabling verbal conversation. Unlike basic letter-based translators, our system understands full gestures, incorporating context to generate natural speech. By bridging the gap between sign language and spoken communication, we’re redefining how the world connects.  

---  

## How We Built It  

Our journey to creating a functional ASL translator involved a phased approach, starting with a conceptual simulation and progressing toward real-time, computer vision-driven recognition.  

### Phase 1: Conceptual Simulation (Text-Based Demo)  
We began with a text-based simulator to establish a foundational understanding of ASL translation. This phase allowed us to:  
- Define ASL signs: We compiled a dictionary of common ASL signs and their textual descriptions.  
- Simulate recognition logic: Implemented functions for letter-by-letter spelling, phrase detection, and sign recognition within a menu-driven interface.  
- Explore user interaction: This approach helped us conceptualize the user experience and refine interaction flow.  

This phase was crucial for solidifying our ASL knowledge and laying the groundwork for real-time implementation.  

### Phase 2: Basic Computer Vision with OpenCV  
Transitioning from the simulation, we integrated OpenCV to enable real-time sign detection:  
- Hand detection and feature extraction: Used contour detection, convex hull analysis, and convexity defect identification to isolate and analyze hand shapes.  
- Sign detection algorithms: Developed methods to recognize specific ASL signs such as "HELLO," "THANK YOU," "YES," and "NO."  
- Text-to-speech integration: Implemented `pyttsx3` to audibly output recognized signs, enhancing accessibility.  
- Real-time processing: Achieved smooth video stream processing and display of recognized signs.  

This phase allowed us to experiment with computer vision techniques and establish a basic real-time translation pipeline.  

### Phase 3: Advanced Hand Tracking with MediaPipe  
To enhance accuracy and robustness, we integrated Google’s MediaPipe for state-of-the-art hand tracking:  
- MediaPipe Hands integration: Leveraged its pre-trained model to extract precise multi-dimensional hand landmark coordinates.  
- Refined sign detection logic: Adapted our algorithms to utilize MediaPipe’s landmark data, improving recognition across hand shapes and orientations.  
- Enhanced movement tracking: Improved detection of dynamic signs like "THANK YOU," "YES," and "NO."  
- Stability and smoothing: Implemented output stabilization and false positive reduction to improve reliability.  

This phase resulted in a highly accurate ASL translator, capable of handling a wide range of gestures and movements.  

### Phase 4: Video Input & Preprocessing  
For seamless real-time recognition, we designed an efficient video input and preprocessing pipeline:  
- Camera Module: A Raspberry Pi camera continuously captures live video as the primary input stream.  
- Remote Processing Server: The raw video feed is transmitted to a remote server for computational efficiency, ensuring minimal latency.  

### Phase 5: Gesture Classification & Speech-to-Text Conversion  
To map recognized signs into meaningful text and speech:  
- Gesture Classification: The extracted landmark data is fed into a trained classification model to interpret hand positions and motion patterns.  
- Text Generation: Predicted gestures are dynamically mapped to linguistic representations, ensuring accurate speech-ready text conversion.  
- Speech-to-Text Integration: The system also captures spoken language via an integrated microphone, utilizing advanced speech-to-text algorithms for additional accessibility.  

---  

## Technologies Used  

- **OpenCV** – for video capture and rendering  
- **MediaPipe** – for hand pose detection and landmark extraction  
- **TensorFlow & Keras** – for training and model prediction using detected landmarks  
- **Tkinter** – GUI library  
- **I3D** – Pre-trained 3D convolutional neural model that was fine-tuned  
- **NumPy** – for numerical operations and array manipulation  
- **Pyttsx3** – for text-to-speech functionality  
- **PIL (Python Imaging Library)** – for image processing  
- **PyTorch** – module for building neural networks  

---  

## Challenges We Faced  

### Lack of Existing Frameworks for Gesture-Based ASL Recognition  
There aren't many established, consistent frameworks for recognizing complex ASL gestures that go beyond letter-based translation. ASL incorporates a variety of hand movements, facial expressions, and spatial positioning, making it challenging to develop a system that accurately interprets full gestures in real time. Existing tools primarily focus on static signs, leaving a significant gap in dynamic, contextual ASL translation.  

### Difficulty in Connecting Camera Output from Raspberry Pi with the Backend  
Integrating the camera module with the backend proved challenging due to latency, bandwidth constraints, and unsupported modules. The Raspberry Pi's processing capabilities were limited, making it difficult to transmit high-quality video in real time to a remote server for analysis. Optimizing this connection to ensure smooth, low-latency communication between the Raspberry Pi and the backend while maintaining video quality was a key hurdle.  

---  

## Accomplishments We're Proud Of  

- Trained and compared several machine learning models, including a traditional CNN, a fine-tuned pre-existing model, and a landmark-based CNN, to evaluate their performance.  
- Built a working prototype that minimized latency, thus functioning in real time.  
- Created a tool that enhances accessibility for the Deaf and Hard of Hearing.  

---  

## What's Next  

- Improve Model Accuracy: Use CNNs or LSTMs for better contextual awareness  
- Sequence-to-Sequence Translation: Recognize full gesture-based sentences  
- User Interface Improvements: Add a clean and intuitive GUI  
- Web-Based Version: Deploy using TensorFlow.js or MediaPipe Web  

---  

## Team  

- **Liam Ellison**  
- **Joel Koshy**  
- **Charan Peeriga**  
- **Areeb Ehsan**  


