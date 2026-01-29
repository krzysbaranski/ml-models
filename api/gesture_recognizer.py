"""
MediaPipe Gesture Recognition Module
Provides gesture recognition functionality using MediaPipe's TFLite model
"""
import os
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from typing import List

# Configure logging
logger = logging.getLogger(__name__)
    
# Model configuration
# Construct path relative to this module's location
_current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_current_dir, "..", "models", "gesture_recognizer.task")


class GestureRecognizer:
    """Wrapper class for MediaPipe gesture recognition"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize the gesture recognizer with the specified model"""
        logger.info(f"Initializing GestureRecognizer with model path: {model_path}")
        self.model_path = model_path
        
        # Ensure model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"The model should be bundled in the Docker image at this path. "
                f"If running locally, ensure the model file exists at the specified path."
            )
        
        # Create a GestureRecognizer object
        logger.info("Creating MediaPipe GestureRecognizer instance...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        logger.info("MediaPipe GestureRecognizer instance created successfully")

    
    def recognize(self, image: np.ndarray) -> vision.GestureRecognizerResult:
        """
        Perform gesture recognition on the given image
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            GestureRecognizerResult containing recognition results
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Perform recognition
        recognition_result = self.recognizer.recognize(mp_image)
        
        return recognition_result
    
    def annotate_image(self, image: np.ndarray, recognition_result: vision.GestureRecognizerResult) -> np.ndarray:
        """
        Draw hand landmarks and gesture labels on the image
        
        Args:
            image: Input image as numpy array
            recognition_result: Recognition results from recognize()
            
        Returns:
            Annotated image as numpy array
        """
        annotated_image = image.copy()
        
        # Get image dimensions
        image_height, image_width, _ = image.shape
        
        # Draw hand landmarks for each detected hand
        if recognition_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                # Draw landmarks
                for landmark in hand_landmarks:
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    # Draw landmark point
                    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
                
                # Draw connections between landmarks
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                    (5, 9), (9, 13), (13, 17)  # Palm
                ]
                
                for connection in connections:
                    start_idx, end_idx = connection
                    start_point = (
                        int(hand_landmarks[start_idx].x * image_width),
                        int(hand_landmarks[start_idx].y * image_height)
                    )
                    end_point = (
                        int(hand_landmarks[end_idx].x * image_width),
                        int(hand_landmarks[end_idx].y * image_height)
                    )
                    cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
                
                # Draw gesture label if available
                if recognition_result.gestures and hand_idx < len(recognition_result.gestures):
                    gesture = recognition_result.gestures[hand_idx][0]  # Top gesture
                    
                    # Get the wrist position (landmark 0) for label placement
                    wrist = hand_landmarks[0]
                    label_x = int(wrist.x * image_width)
                    label_y = int(wrist.y * image_height) - 20
                    
                    # Draw gesture label
                    label = f"{gesture.category_name} ({gesture.score:.2f})"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_image,
                        (label_x, label_y - label_size[1] - 5),
                        (label_x + label_size[0], label_y + 5),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_image,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                    )
        
        return annotated_image
    
    def format_recognition_results(self, recognition_result: vision.GestureRecognizerResult) -> List[dict]:
        """
        Format recognition results as a list of dictionaries
        
        Args:
            recognition_result: Recognition results from recognize()
            
        Returns:
            List of recognition dictionaries with gesture, score, handedness, and landmarks info
        """
        results = []
        
        # Check if any gestures were detected
        if not recognition_result.gestures:
            return results
        
        # Process each detected hand
        for hand_idx in range(len(recognition_result.gestures)):
            result = {}
            
            # Get gesture information
            if recognition_result.gestures[hand_idx]:
                top_gesture = recognition_result.gestures[hand_idx][0]
                result["gesture"] = top_gesture.category_name
                result["score"] = float(top_gesture.score)
            
            # Get handedness (left or right hand)
            if recognition_result.handedness and hand_idx < len(recognition_result.handedness):
                handedness = recognition_result.handedness[hand_idx][0]
                result["handedness"] = handedness.category_name
                result["handedness_score"] = float(handedness.score)
            
            # Get hand landmarks
            if recognition_result.hand_landmarks and hand_idx < len(recognition_result.hand_landmarks):
                landmarks = recognition_result.hand_landmarks[hand_idx]
                result["landmarks"] = [
                    {
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z)
                    }
                    for landmark in landmarks
                ]
            
            results.append(result)
        
        return results
