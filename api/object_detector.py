"""
MediaPipe Object Detection Module
Provides object detection functionality using MediaPipe's TFLite model
"""
import os
import urllib.request
import textwrap
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from typing import List


# Model configuration
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
MODEL_PATH = "models/efficientdet_lite0.tflite"


class ObjectDetector:
    """Wrapper class for MediaPipe object detection"""
    
    def __init__(self, model_path: str = MODEL_PATH):
        """Initialize the object detector with the specified model"""
        self.model_path = model_path
        self._ensure_model_exists()
        
        # Create an ObjectDetector object
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            max_results=10
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
    
    def _ensure_model_exists(self):
        """Download the model if it doesn't exist"""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"Downloading model from {MODEL_URL}...")
            
            try:
                # Try to download with requests library if available
                try:
                    import requests
                    response = requests.get(MODEL_URL, timeout=30)
                    response.raise_for_status()
                    with open(self.model_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Model downloaded to {self.model_path}")
                except ImportError:
                    # Fall back to urllib with headers
                    req = urllib.request.Request(
                        MODEL_URL,
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    
                    with urllib.request.urlopen(req, timeout=30) as response:
                        with open(self.model_path, 'wb') as out_file:
                            out_file.write(response.read())
                    
                    print(f"Model downloaded to {self.model_path}")
            except Exception as e:
                error_msg = textwrap.dedent(f"""
                    Failed to download model: {str(e)}
                    
                    Please manually download the model from:
                    {MODEL_URL}
                    
                    And save it to:
                    {self.model_path}
                    
                    Or run:
                    wget -O {self.model_path} {MODEL_URL}
                """)
                raise RuntimeError(error_msg)
    
    def detect(self, image: np.ndarray) -> vision.ObjectDetectorResult:
        """
        Perform object detection on the given image
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            ObjectDetectorResult containing detection results
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Perform detection
        detection_result = self.detector.detect(mp_image)
        
        return detection_result
    
    def annotate_image(self, image: np.ndarray, detection_result: vision.ObjectDetectorResult) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image
        
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
            
            # Get category and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            
            # Draw label
            label = f"{category_name} ({probability})"
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
        
        return annotated_image
    
    def format_detection_results(self, detection_result: vision.ObjectDetectorResult) -> List[dict]:
        """
        Format detection results as a list of dictionaries
        
        Args:
            detection_result: Detection results from detect()
            
        Returns:
            List of detection dictionaries with category, score, and bbox info
        """
        results = []
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            
            result = {
                "category": category.category_name,
                "score": float(category.score),
                "bbox": {
                    "x": bbox.origin_x,
                    "y": bbox.origin_y,
                    "width": bbox.width,
                    "height": bbox.height
                }
            }
            results.append(result)
        
        return results
