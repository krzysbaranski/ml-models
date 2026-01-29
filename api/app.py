"""
FastAPI application for object detection on image frames
Suitable for integration with home security camera systems
"""
import base64
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np
from object_detector import ObjectDetector
from face_detector import FaceDetector
from gesture_recognizer import GestureRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Home Security Camera Object Detection API",
    description="API service for object detection on single image frames using MediaPipe",
    version="1.0.0"
)

# Mount static files
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Initialize the object detector
logger.info("Initializing object detector...")
detector = ObjectDetector()
logger.info("Object detector initialized successfully")

# Initialize the face detector
logger.info("Initializing face detector...")
face_detector = FaceDetector()
logger.info("Face detector initialized successfully")

# Initialize the gesture recognizer
try:
    logger.info("Initializing gesture recognizer...")
    gesture_recognizer = GestureRecognizer()
    logger.info("Gesture recognizer initialized successfully")
    GESTURE_RECOGNIZER_AVAILABLE = True
except FileNotFoundError as e:
    logger.warning(f"Gesture recognizer model not found: {e}")
    logger.warning("Gesture recognition endpoints will not be available")
    logger.warning("To enable gesture recognition, download the model from:")
    logger.warning("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
    logger.warning("And place it in the models/ directory")
    gesture_recognizer = None
    GESTURE_RECOGNIZER_AVAILABLE = False
except Exception as e:
    logger.error(f"Failed to initialize gesture recognizer: {e}")
    gesture_recognizer = None
    GESTURE_RECOGNIZER_AVAILABLE = False


# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses"""
    start_time = time.time()
    
    # Log request - only path without query parameters
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response


@app.get("/")
async def root():
    """Root endpoint with API information"""
    endpoints = {
        "/detect": "POST - Upload an image for object detection",
        "/detect/image": "POST - Upload an image and get annotated image only",
        "/detect_faces": "POST - Upload an image for face detection",
        "/detect_faces/image": "POST - Upload an image and get annotated image with faces only",
        "/upload": "GET - Web interface for uploading images",
        "/health": "GET - Health check endpoint"
    }
    
    # Add gesture recognition endpoints only if available
    if GESTURE_RECOGNIZER_AVAILABLE:
        endpoints["/recognize_gesture"] = "POST - Upload an image for gesture recognition"
        endpoints["/recognize_gesture/image"] = "POST - Upload an image and get annotated image with gestures only"
    
    return {
        "service": "Home Security Camera Object Detection API",
        "version": "1.0.0",
        "endpoints": endpoints
    }


@app.get("/upload")
async def upload_page():
    """Serve the image upload web interface"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "upload.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        raise HTTPException(status_code=404, detail="Upload page not found")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    return_image: bool = True,
    image_format: str = "jpeg"
):
    """
    Detect objects in an uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        return_image: Whether to return annotated image (default: True)
        image_format: Format for returned image - 'jpeg' or 'png' (default: 'jpeg')
        
    Returns:
        JSON response with detection results and optionally annotated image
    """
    logger.info(f"Object detection request received - File: {file.filename}, Return image: {return_image}, Format: {image_format}")
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Invalid image file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Image loaded successfully - Shape: {image.shape}")
        
        # Perform object detection
        detection_result = detector.detect(image)
        
        # Format results
        detections = detector.format_detection_results(detection_result)
        logger.info(f"Detection complete - Found {len(detections)} objects")
        
        response_data = {
            "detections": detections,
            "count": len(detections)
        }
        
        # If return_image is True, return annotated image
        if return_image:
            annotated_image = detector.annotate_image(image, detection_result)
            
            # Encode image
            if image_format.lower() == "png":
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                ext = ".png"
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                ext = ".jpg"
            
            _, buffer = cv2.imencode(ext, annotated_image, encode_param)
            image_bytes = buffer.tobytes()
            
            # Return JSON with base64 encoded image
            response_data["annotated_image"] = base64.b64encode(image_bytes).decode('utf-8')
            response_data["image_format"] = image_format
            logger.debug(f"Annotated image added to response - Size: {len(image_bytes)} bytes")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.exception(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect/image")
async def detect_objects_image_only(
    file: UploadFile = File(...),
    image_format: str = "jpeg"
):
    """
    Detect objects and return only the annotated image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        image_format: Format for returned image - 'jpeg' or 'png' (default: 'jpeg')
        
    Returns:
        Annotated image with bounding boxes
    """
    logger.info(f"Image-only detection request received - File: {file.filename}, Format: {image_format}")
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Invalid image file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Image loaded successfully - Shape: {image.shape}")
        
        # Perform object detection
        detection_result = detector.detect(image)
        
        # Get detection count for logging
        detection_count = len(detection_result.detections)
        logger.info(f"Detection complete - Found {detection_count} objects")
        
        # Annotate image
        annotated_image = detector.annotate_image(image, detection_result)
        
        # Encode image
        if image_format.lower() == "png":
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            ext = ".png"
            media_type = "image/png"
        else:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ext = ".jpg"
            media_type = "image/jpeg"
        
        _, buffer = cv2.imencode(ext, annotated_image, encode_param)
        image_bytes = buffer.tobytes()
        
        logger.debug(f"Returning annotated image - Size: {len(image_bytes)} bytes, Type: {media_type}")
        return Response(content=image_bytes, media_type=media_type)
        
    except Exception as e:
        logger.exception(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect_faces")
async def detect_faces(
    file: UploadFile = File(...),
    return_image: bool = True,
    image_format: str = "jpeg"
):
    """
    Detect faces in an uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        return_image: Whether to return annotated image (default: True)
        image_format: Format for returned image - 'jpeg' or 'png' (default: 'jpeg')
        
    Returns:
        JSON response with detection results and optionally annotated image
    """
    logger.info(f"Face detection request received - File: {file.filename}, Return image: {return_image}, Format: {image_format}")
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Invalid image file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Image loaded successfully - Shape: {image.shape}")
        
        # Perform face detection
        detection_result = face_detector.detect(image)
        
        # Format results
        detections = face_detector.format_detection_results(detection_result)
        logger.info(f"Detection complete - Found {len(detections)} faces")
        
        response_data = {
            "detections": detections,
            "count": len(detections)
        }
        
        # If return_image is True, return annotated image
        if return_image:
            annotated_image = face_detector.annotate_image(image, detection_result)
            
            # Encode image
            if image_format.lower() == "png":
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                ext = ".png"
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                ext = ".jpg"
            
            _, buffer = cv2.imencode(ext, annotated_image, encode_param)
            image_bytes = buffer.tobytes()
            
            # Return JSON with base64 encoded image
            response_data["annotated_image"] = base64.b64encode(image_bytes).decode('utf-8')
            response_data["image_format"] = image_format
            logger.debug(f"Annotated image added to response - Size: {len(image_bytes)} bytes")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.exception(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/detect_faces/image")
async def detect_faces_image_only(
    file: UploadFile = File(...),
    image_format: str = "jpeg"
):
    """
    Detect faces and return only the annotated image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        image_format: Format for returned image - 'jpeg' or 'png' (default: 'jpeg')
        
    Returns:
        Annotated image with bounding boxes around faces
    """
    logger.info(f"Image-only face detection request received - File: {file.filename}, Format: {image_format}")
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Invalid image file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Image loaded successfully - Shape: {image.shape}")
        
        # Perform face detection
        detection_result = face_detector.detect(image)
        
        # Get detection count for logging
        detection_count = len(detection_result.detections)
        logger.info(f"Detection complete - Found {detection_count} faces")
        
        # Annotate image
        annotated_image = face_detector.annotate_image(image, detection_result)
        
        # Encode image
        if image_format.lower() == "png":
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            ext = ".png"
            media_type = "image/png"
        else:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ext = ".jpg"
            media_type = "image/jpeg"
        
        _, buffer = cv2.imencode(ext, annotated_image, encode_param)
        image_bytes = buffer.tobytes()
        
        logger.debug(f"Returning annotated image - Size: {len(image_bytes)} bytes, Type: {media_type}")
        return Response(content=image_bytes, media_type=media_type)
        
    except Exception as e:
        logger.exception(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/recognize_gesture")
async def recognize_gesture(
    file: UploadFile = File(...),
    return_image: bool = True,
    image_format: str = "jpeg"
):
    """
    Recognize hand gestures in an uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        return_image: Whether to return annotated image (default: True)
        image_format: Format for returned image - 'jpeg' or 'png' (default: 'jpeg')
        
    Returns:
        JSON response with recognition results and optionally annotated image
    """
    if not GESTURE_RECOGNIZER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Gesture recognition is not available. Model file not found. "
                   "Please download gesture_recognizer.task and place it in the models/ directory."
        )
    
    logger.info(f"Gesture recognition request received - File: {file.filename}, Return image: {return_image}, Format: {image_format}")
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Invalid image file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Image loaded successfully - Shape: {image.shape}")
        
        # Perform gesture recognition
        recognition_result = gesture_recognizer.recognize(image)
        
        # Format results
        recognitions = gesture_recognizer.format_recognition_results(recognition_result)
        logger.info(f"Recognition complete - Found {len(recognitions)} hand(s)")
        
        response_data = {
            "recognitions": recognitions,
            "count": len(recognitions)
        }
        
        # If return_image is True, return annotated image
        if return_image:
            annotated_image = gesture_recognizer.annotate_image(image, recognition_result)
            
            # Encode image
            if image_format.lower() == "png":
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                ext = ".png"
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                ext = ".jpg"
            
            _, buffer = cv2.imencode(ext, annotated_image, encode_param)
            image_bytes = buffer.tobytes()
            
            # Return JSON with base64 encoded image
            response_data["annotated_image"] = base64.b64encode(image_bytes).decode('utf-8')
            response_data["image_format"] = image_format
            logger.debug(f"Annotated image added to response - Size: {len(image_bytes)} bytes")
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.exception(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/recognize_gesture/image")
async def recognize_gesture_image_only(
    file: UploadFile = File(...),
    image_format: str = "jpeg"
):
    """
    Recognize gestures and return only the annotated image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        image_format: Format for returned image - 'jpeg' or 'png' (default: 'jpeg')
        
    Returns:
        Annotated image with hand landmarks and gesture labels
    """
    if not GESTURE_RECOGNIZER_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Gesture recognition is not available. Model file not found. "
                   "Please download gesture_recognizer.task and place it in the models/ directory."
        )
    
    logger.info(f"Image-only gesture recognition request received - File: {file.filename}, Format: {image_format}")
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Invalid image file uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        logger.info(f"Image loaded successfully - Shape: {image.shape}")
        
        # Perform gesture recognition
        recognition_result = gesture_recognizer.recognize(image)
        
        # Get recognition count for logging
        recognition_count = len(recognition_result.gestures) if recognition_result.gestures else 0
        logger.info(f"Recognition complete - Found {recognition_count} hand(s)")
        
        # Annotate image
        annotated_image = gesture_recognizer.annotate_image(image, recognition_result)
        
        # Encode image
        if image_format.lower() == "png":
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            ext = ".png"
            media_type = "image/png"
        else:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            ext = ".jpg"
            media_type = "image/jpeg"
        
        _, buffer = cv2.imencode(ext, annotated_image, encode_param)
        image_bytes = buffer.tobytes()
        
        logger.debug(f"Returning annotated image - Size: {len(image_bytes)} bytes, Type: {media_type}")
        return Response(content=image_bytes, media_type=media_type)
        
    except Exception as e:
        logger.exception(f"Error processing image {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Home Security Camera Object Detection API on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
