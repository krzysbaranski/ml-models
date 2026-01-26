"""
FastAPI application for object detection on image frames
Suitable for integration with home security camera systems
"""
import base64
import logging
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np
from object_detector import ObjectDetector

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

# Initialize the object detector
logger.info("Initializing object detector...")
detector = ObjectDetector()
logger.info("Object detector initialized successfully")


# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests and responses"""
    start_time = time.time()
    
    # Log request
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
    logger.debug("Root endpoint called")
    return {
        "service": "Home Security Camera Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - Upload an image for object detection",
            "/health": "GET - Health check endpoint"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    logger.debug("Health check endpoint called")
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


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Home Security Camera Object Detection API on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
