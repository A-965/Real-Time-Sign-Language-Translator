# feature_extractor.py
import numpy as np
import torch

class FeatureExtractor:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        
    def extract_features(self, landmarks):
        """
        Extract features from hand landmarks
        Args:
            landmarks: numpy array of shape (21, 3) for single hand
        Returns:
            feature vector
        """
        if landmarks is None or len(landmarks) == 0:
            return np.zeros(63)  # 21 landmarks * 3 coordinates
        
        # Flatten landmarks
        features = landmarks.flatten()
        
        # Normalize relative to wrist (landmark 0)
        if len(landmarks.shape) == 2 and landmarks.shape[0] >= 21:
            wrist = landmarks[0]
            normalized_landmarks = landmarks - wrist
            features = normalized_landmarks.flatten()
        
        # Calculate distances between key points
        distances = self._calculate_distances(landmarks)
        
        # Calculate angles between fingers
        angles = self._calculate_angles(landmarks)
        
        # Combine all features
        combined_features = np.concatenate([features, distances, angles])
        
        return combined_features
    
    def _calculate_distances(self, landmarks):
        """Calculate distances between key landmarks"""
        if landmarks.shape[0] < 21:
            return np.zeros(10)
        
        distances = []
        # Distance from wrist to fingertips
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        wrist = landmarks[0]
        
        for tip in fingertips:
            dist = np.linalg.norm(landmarks[tip] - wrist)
            distances.append(dist)
        
        # Distance between fingertips
        for i in range(len(fingertips)-1):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                distances.append(dist)
        
        return np.array(distances[:10])  # Limit to 10 distances
    
    def _calculate_angles(self, landmarks):
        """Calculate angles between finger segments"""
        if landmarks.shape[0] < 21:
            return np.zeros(5)
        
        angles = []
        # Calculate angles for each finger
        finger_indices = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16], # Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for finger in finger_indices:
            if len(finger) >= 3:
                v1 = landmarks[finger[1]] - landmarks[finger[0]]
                v2 = landmarks[finger[2]] - landmarks[finger[1]]
                
                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
        
        return np.array(angles)