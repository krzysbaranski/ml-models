"""
FastAPI application for object detection on image frames
Suitable for integration with home security camera systems
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np
from object_detector import ObjectDetector
import io


app = FastAPI(
    title="Home Security Camera Object Detection API",
    description="API service for object detection on single image frames using MediaPipe",
    version="1.0.0"
)

# Initialize the object detector
detector = ObjectDetector()


@app.get("/")
async def root():
    """Root endpoint with API information"""
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
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform object detection
        detection_result = detector.detect(image)
        
        # Format results
        detections = detector.format_detection_results(detection_result)
        
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
                media_type = "image/png"
            else:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                ext = ".jpg"
                media_type = "image/jpeg"
            
            _, buffer = cv2.imencode(ext, annotated_image, encode_param)
            image_bytes = buffer.tobytes()
            
            # Return multipart response with JSON header and image
            # For simplicity, we'll return JSON with base64 encoded image
            import base64
            response_data["annotated_image"] = base64.b64encode(image_bytes).decode('utf-8')
            response_data["image_format"] = image_format
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
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
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Perform object detection
        detection_result = detector.detect(image)
        
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
        
        return Response(content=image_bytes, media_type=media_type)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
