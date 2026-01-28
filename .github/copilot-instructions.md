# GitHub Copilot Instructions for ml-models

## Project Overview

This repository contains a Home Security Camera Object Detection API built with FastAPI and MediaPipe. It provides a standalone API service for object detection on single image frames, suitable for integration with home security camera systems.

## Technology Stack

- **Language**: Python 3.x
- **Web Framework**: FastAPI 0.109.1
- **ML Framework**: MediaPipe 0.10.14
- **Computer Vision**: OpenCV (cv2) 4.9.0.80
- **Model**: EfficientDet Lite0 TFLite
- **Server**: Uvicorn 0.27.0
- **Containerization**: Docker

## Project Structure

```
ml-models/
├── api/
│   ├── app.py              # FastAPI application with HTTP endpoints
│   ├── object_detector.py  # MediaPipe object detection wrapper
│   └── static/             # Static files for web interface
├── examples/               # Example usage scripts
├── models/                 # TFLite model files (auto-downloaded)
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── download_model.sh       # Script to manually download model
```

## Coding Standards and Conventions

### Python Code Style

- **Docstrings**: Use triple-quoted docstrings for all modules, classes, and functions
- **Type Hints**: Use type hints for function parameters and return values (e.g., `image: np.ndarray`, `-> List[dict]`)
- **Imports**: Group imports logically - standard library, third-party, local modules
- **Logging**: Use the `logging` module with appropriate levels (INFO, ERROR, DEBUG)
- **Error Handling**: Use try-except blocks with specific exception types and informative error messages

### Code Organization

- **Functions**: Keep functions focused on a single responsibility
- **Classes**: Use classes to encapsulate related functionality (e.g., `ObjectDetector`)
- **Constants**: Define module-level constants in UPPERCASE (e.g., `MODEL_PATH`)
- **Comments**: Write clear comments explaining the "why" not the "what"

### FastAPI Specific

- **Endpoints**: Use async functions for all route handlers
- **Request Models**: Use Pydantic models or standard types for request validation
- **Response Models**: Return JSONResponse or Response objects
- **Documentation**: Include docstrings for all endpoints explaining parameters and return values
- **Middleware**: Use middleware for cross-cutting concerns like logging

### MediaPipe Integration

- **Image Format**: Convert OpenCV BGR images to RGB before passing to MediaPipe
- **Model Loading**: Load models once during initialization, not per request
- **Detection Results**: Use MediaPipe's result objects and convert to simpler dictionaries for API responses

## Build, Test, and Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API locally
cd api
python app.py
```

### Docker

```bash
# Build Docker image
docker build -t object-detection-api .

# Run with Docker
docker run -p 8000:8000 object-detection-api

# Run with Docker Compose
docker-compose up
```

### Testing

```bash
# Test API endpoints
./examples/test_api.sh test_image.jpg

# Test with curl
curl -X POST "http://localhost:8000/detect" -F "file=@test_image.jpg"

# Access web interface
http://localhost:8000/upload
```

## Key Components

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /upload` - Web interface for testing
- `POST /detect` - Object detection with JSON response
- `POST /detect/image` - Object detection with image response

### Object Detection

- **Model**: EfficientDet Lite0 (4MB TFLite model)
- **Input**: RGB images
- **Output**: Object categories, confidence scores, bounding boxes
- **Categories**: COCO dataset (person, car, dog, cat, etc.)
- **Score Threshold**: 0.5 (configurable in object_detector.py)
- **Max Results**: 10 detections per image

## Best Practices for This Repository

### When Adding New Features

1. **Maintain API Compatibility**: Don't break existing endpoints
2. **Add Logging**: Log important operations with appropriate levels
3. **Error Handling**: Always handle exceptions and return meaningful HTTP status codes
4. **Documentation**: Update README.md with new endpoints or features
5. **Type Safety**: Use type hints for all function parameters and returns

### When Modifying Detection Logic

1. **Keep ObjectDetector Focused**: All MediaPipe logic should be in `object_detector.py`
2. **Preserve Image Formats**: Handle BGR/RGB conversions properly
3. **Test with Sample Images**: Verify detection quality after changes
4. **Maintain Performance**: Detection should stay under 200ms per image

### When Working with Docker

1. **Model Bundling**: Ensure model files are included in Docker image or auto-downloaded
2. **Port Exposure**: API should run on port 8000
3. **Environment Variables**: Use env vars for configuration, not hardcoded values
4. **Layer Optimization**: Optimize Dockerfile layers for faster builds

### Security Considerations

1. **Input Validation**: Validate uploaded files are valid images
2. **File Size Limits**: Consider implementing file size limits for uploads
3. **Resource Management**: Clean up temporary files and images after processing
4. **Logging**: Don't log sensitive information or uploaded image content

### Integration Guidelines

1. **Home Assistant**: API is designed to be compatible with Home Assistant integrations
2. **Security Cameras**: Supports continuous frame processing from camera feeds
3. **Response Format**: JSON responses include detection arrays with category, score, and bbox
4. **Image Options**: Support both base64-encoded images in JSON and raw image responses

## CI/CD

- **GitHub Actions**: Automated Docker builds on push to main/master branches
- **Docker Hub**: Images automatically published with version tags
- **Secrets Required**: `DOCKER_USERNAME` and `DOCKER_PASSWORD` for Docker Hub publishing

## Dependencies Management

- Keep `requirements.txt` up to date with pinned versions for stability
- Test compatibility when updating major dependencies (FastAPI, MediaPipe, OpenCV)
- Use `pip install -r requirements.txt` for consistent environments

## Common Tasks

### Add a New API Endpoint

1. Define route handler in `api/app.py` as async function
2. Add type hints for parameters and return value
3. Include docstring with endpoint description
4. Add logging for request and response
5. Update README.md with endpoint documentation

### Modify Detection Parameters

Edit `api/object_detector.py`:
```python
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,  # Adjust minimum confidence
    max_results=10        # Adjust max detections
)
```

### Add Example Scripts

Place in `examples/` directory with:
- Clear filename describing purpose
- Docstring explaining usage
- Example command in README.md

## Resources

- **MediaPipe Documentation**: https://developers.google.com/mediapipe
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Model Source**: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/
