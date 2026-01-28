"""
MediaPipe Face Detection Module
Provides face detection functionality using MediaPipe's TFLite model
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
MODEL_PATH = os.path.join(_current_dir, "..", "models", "blaze_face_short_range.tflite")


class FaceDetector:
    """Wrapper class for MediaPipe face detection"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize the face detector with the specified model"""
        logger.info(f"Initializing FaceDetector with model path: {model_path}")
        self.model_path = model_path
        
        # Ensure model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                f"The model should be bundled in the Docker image at this path. "
                f"If running locally, ensure the model file exists at the specified path."
            )
        
        # Create a FaceDetector object
        logger.info("Creating MediaPipe FaceDetector instance...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        logger.info("MediaPipe FaceDetector instance created successfully")

    
    def detect(self, image: np.ndarray) -> vision.FaceDetectorResult:
        """
        Perform face detection on the given image
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            FaceDetectorResult containing detection results
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Perform detection
        detection_result = self.detector.detect(mp_image)
        
        return detection_result
    
    def annotate_image(self, image: np.ndarray, detection_result: vision.FaceDetectorResult) -> np.ndarray:
        """
        Draw bounding boxes and keypoints on the image
        
        Args:
            image: Input image as numpy array
            detection_result: Detection results from detect()
            
        Returns:
            Annotated image as numpy array
        """
        annotated_image = image.copy()
        
        for detection in detection_result.detections:
            # Get bounding box coordinates
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, start_point, end_point, (0, 255, 0), 2)
            
            # Get confidence score
            if detection.categories:
                category = detection.categories[0]
                probability = round(category.score, 2)
                
                # Draw label
                label = f"Face ({probability})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(bbox.origin_y - 10, label_size[1])
                
                # Draw label background
                cv2.rectangle(
                    annotated_image,
                    (bbox.origin_x, label_y - label_size[1] - 5),
                    (bbox.origin_x + label_size[0], label_y + 5),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_image,
                    label,
                    (bbox.origin_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Draw keypoints if available
            if hasattr(detection, 'keypoints') and detection.keypoints:
                for keypoint in detection.keypoints:
                    keypoint_px = (int(keypoint.x * image.shape[1]), 
                                 int(keypoint.y * image.shape[0]))
                    cv2.circle(annotated_image, keypoint_px, 3, (255, 0, 0), -1)
        
        return annotated_image
    
    def format_detection_results(self, detection_result: vision.FaceDetectorResult) -> List[dict]:
        """
        Format detection results as a list of dictionaries
        
        Args:
            detection_result: Detection results from detect()
            
        Returns:
            List of detection dictionaries with score, bbox, and keypoints info
        """
        results = []
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            
            result = {
                "bbox": {
                    "x": bbox.origin_x,
                    "y": bbox.origin_y,
                    "width": bbox.width,
                    "height": bbox.height
                }
            }
            
            # Add confidence score if available
            if detection.categories:
                category = detection.categories[0]
                result["score"] = float(category.score)
            
            # Add keypoints if available
            if hasattr(detection, 'keypoints') and detection.keypoints:
                keypoints = []
                for keypoint in detection.keypoints:
                    keypoints.append({
                        "x": float(keypoint.x),
                        "y": float(keypoint.y)
                    })
                result["keypoints"] = keypoints
            
            results.append(result)
        
        return results
