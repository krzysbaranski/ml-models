# Home Security Camera Object Detection API

A standalone API service for object detection on single image frames, suitable for integration with home security camera systems. Built with FastAPI and MediaPipe, this service can process camera frames and return detected objects with bounding boxes.

## Features

- 🎯 **Object Detection**: Uses MediaPipe's EfficientDet Lite0 TFLite model for fast and accurate detection
- 🖼️ **Image Annotation**: Returns annotated images with bounding boxes and labels
- 🐳 **Docker Ready**: Fully containerized for easy deployment
- 🔄 **Auto-Download**: Automatically downloads the model on first run
- 📡 **REST API**: Simple HTTP endpoints for easy integration
- 🏠 **IoT Compatible**: Perfect for home automation and security camera systems

## Quick Start

### Option 1: Run with Docker (Recommended)

1. **Build the Docker image:**
```bash
docker build -t object-detection-api .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 object-detection-api
```

The API will be available at `http://localhost:8000`

### Option 2: Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the API:**
```bash
cd api
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "service": "Home Security Camera Object Detection API",
  "version": "1.0.0",
  "endpoints": {
    "/detect": "POST - Upload an image for object detection",
    "/health": "GET - Health check endpoint"
  }
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `POST /detect`
Main endpoint for object detection. Accepts an image file and returns detection results with optional annotated image.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `return_image` (optional): Whether to return annotated image (default: true)
- `image_format` (optional): Format for returned image - 'jpeg' or 'png' (default: 'jpeg')

**Example Request:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@path/to/your/image.jpg" \
  -F "return_image=true" \
  -F "image_format=jpeg"
```

**Example Response:**
```json
{
  "detections": [
    {
      "category": "person",
      "score": 0.87,
      "bbox": {
        "x": 120,
        "y": 80,
        "width": 200,
        "height": 350
      }
    },
    {
      "category": "car",
      "score": 0.92,
      "bbox": {
        "x": 400,
        "y": 200,
        "width": 300,
        "height": 250
      }
    }
  ],
  "count": 2,
  "annotated_image": "base64_encoded_image_data...",
  "image_format": "jpeg"
}
```

### `POST /detect/image`
Returns only the annotated image (no JSON response).

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `image_format` (optional): Format for returned image - 'jpeg' or 'png' (default: 'jpeg')

**Example Request:**
```bash
curl -X POST "http://localhost:8000/detect/image" \
  -F "file=@path/to/your/image.jpg" \
  -F "image_format=jpeg" \
  --output annotated_image.jpg
```

## Usage Examples

### Python Client Example

```python
import requests

# Send image for detection
url = "http://localhost:8000/detect"
files = {"file": open("camera_frame.jpg", "rb")}
params = {"return_image": True, "image_format": "jpeg"}

response = requests.post(url, files=files, data=params)
result = response.json()

print(f"Detected {result['count']} objects:")
for detection in result['detections']:
    print(f"  - {detection['category']}: {detection['score']:.2f}")

# Save annotated image
if "annotated_image" in result:
    import base64
    image_data = base64.b64decode(result["annotated_image"])
    with open("annotated_output.jpg", "wb") as f:
        f.write(image_data)
```

### Home Security Camera Integration

For integration with home security cameras (e.g., using Home Assistant, Frigate, or similar systems):

1. **Continuous Frame Processing:**
```python
import requests
import time

def process_camera_frame(frame_path):
    """Process a single camera frame"""
    url = "http://localhost:8000/detect"
    files = {"file": open(frame_path, "rb")}
    
    response = requests.post(url, files=files)
    return response.json()

# Example: Process frames every 2 seconds
while True:
    result = process_camera_frame("latest_frame.jpg")
    
    # Trigger actions based on detections
    for detection in result["detections"]:
        if detection["category"] == "person" and detection["score"] > 0.8:
            print("Person detected! Sending alert...")
            # Add your alert logic here
    
    time.sleep(2)
```

2. **Home Assistant Integration Example:**
```yaml
# configuration.yaml
rest_command:
  detect_objects:
    url: http://localhost:8000/detect
    method: POST
    content_type: multipart/form-data
    payload: file
```

### curl Examples

**Get API info:**
```bash
curl http://localhost:8000/
```

**Detect objects (JSON response with base64 image):**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg" \
  | jq '.detections'
```

**Get only annotated image:**
```bash
curl -X POST "http://localhost:8000/detect/image" \
  -F "file=@test_image.jpg" \
  --output result.jpg
```

**Detect without returning image (faster):**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg" \
  -F "return_image=false"
```

## Architecture

```
ml-models/
├── api/
│   ├── app.py              # FastAPI application
│   └── object_detector.py  # MediaPipe object detection wrapper
├── models/                 # Model files (auto-downloaded)
│   └── efficientdet_lite0.tflite
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Model Information

- **Model**: EfficientDet Lite0
- **Framework**: TensorFlow Lite (via MediaPipe)
- **Input**: RGB images
- **Output**: Object detections with categories, confidence scores, and bounding boxes
- **Categories**: COCO dataset categories (person, car, dog, cat, etc.)
- **Auto-download**: The model is automatically downloaded on first run from MediaPipe's official repository

## Configuration

You can modify the detection parameters in `api/object_detector.py`:

```python
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,  # Minimum confidence score
    max_results=10        # Maximum number of detections
)
```

## Performance

- **Model Size**: ~4MB
- **Inference Time**: ~100-200ms per image (depending on image size and hardware)
- **Recommended**: Use GPU-enabled containers for better performance

## Troubleshooting

### Docker build fails
- Ensure you have enough disk space
- Check that Docker is running properly

### Model download fails
- Check internet connectivity
- Verify firewall settings allow access to `storage.googleapis.com`

### Poor detection quality
- Adjust `score_threshold` in the detector options
- Ensure images are well-lit and in focus
- Try different camera angles

## License

This project uses MediaPipe which is licensed under Apache License 2.0.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
