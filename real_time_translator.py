# real_time_translator.py
import cv2
import torch
import numpy as np
from collections import deque
import time

from feature_extractor import FeatureExtractor
from hand_detector import HandDetector
from model import SignLanguageClassifier

class RealTimeSignLanguageTranslator:
    def __init__(self, model_path, class_names, sequence_length=30, confidence_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize components
        self.hand_detector = HandDetector()
        self.feature_extractor = FeatureExtractor()
        
        # Sequence buffer for temporal modeling
        self.sequence_buffer = deque(maxlen=sequence_length)
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        
    def _load_model(self, model_path):
        # Determine input size based on your feature extraction
        input_size = 78  # Adjust based on your feature extraction
        hidden_size = 128
        num_classes = len(self.class_names)
        
        model = SignLanguageClassifier(input_size, hidden_size, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_frame(self, frame):
        """Process a single frame and return prediction"""
        # Detect hand landmarks
        landmarks = self.hand_detector.detect_landmarks(frame)
        
        if landmarks is not None and len(landmarks) > 0:
            # Use the first detected hand
            hand_landmarks = landmarks[0]
            
            # Extract features
            features = self.feature_extractor.extract_features(hand_landmarks)
            
            # Add to sequence buffer
            self.sequence_buffer.append(features)
            
            # Make prediction if we have enough frames
            if len(self.sequence_buffer) == self.sequence_length:
                sequence = np.array(list(self.sequence_buffer))
                sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(sequence_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    confidence = confidence.item()
                    predicted_class = predicted_class.item()
                    
                    # Add to prediction buffer for smoothing
                    if confidence > self.confidence_threshold:
                        self.prediction_buffer.append(predicted_class)
                    
                    # Get smoothed prediction
                    if len(self.prediction_buffer) > 0:
                        # Most common prediction in buffer
                        smoothed_prediction = max(set(self.prediction_buffer), 
                                                key=self.prediction_buffer.count)
                        return self.class_names[smoothed_prediction], confidence
        
        return "No sign detected", 0.0
    
    def run_real_time(self):
        """Run real-time sign language translation"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Make prediction
            prediction, confidence = self.predict_frame(frame)
            
            # Draw hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hand_detector.hands.process(rgb_frame)
            frame = self.hand_detector.draw_landmarks(frame, results)
            
            # Display prediction
            cv2.putText(frame, f"Sign: {prediction}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = end_time
                
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display frame
            cv2.imshow('Real-Time Sign Language Translator', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()