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

### Option 1: Pull from Docker Hub (Recommended)

```bash
# Replace <your-dockerhub-username> with the actual Docker Hub username
docker pull <your-dockerhub-username>/ml-models:latest
docker run -p 8000:8000 <your-dockerhub-username>/ml-models:latest
```

The API will be available at `http://localhost:8000`

The model will be automatically downloaded on first run.

### Option 2: Run with Docker Compose (Easiest for Development)

```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

The model will be automatically downloaded on first run.

### Option 3: Build Docker Image Locally

1. **Build the Docker image:**
```bash
docker build -t object-detection-api .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 object-detection-api
```

The API will be available at `http://localhost:8000`

### Option 4: Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **(Optional) Pre-download the model:**
```bash
./download_model.sh
```

Or manually download from: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite
and save to `models/efficientdet_lite0.tflite`

3. **Run the API:**
```bash
cd api
python app.py
```

The API will be available at `http://localhost:8000`

**Note:** The model will be automatically downloaded on first run if not already present.

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
    "/detect/image": "POST - Upload an image and get annotated image only",
    "/upload": "GET - Web interface for uploading images",
    "/health": "GET - Health check endpoint"
  }
}
```

### `GET /upload`
Web interface for uploading and processing images with a visual interface.

**Usage:**
Open `http://localhost:8000/upload` in your browser to access a user-friendly web form where you can:
- Upload images via drag-and-drop or file selection
- View the original image
- See the annotated image with detected objects
- Process multiple images easily

This is the easiest way to test the object detection API interactively.

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

### Web Interface (Easiest)

Simply open your browser and navigate to:
```
http://localhost:8000/upload
```

This provides a visual interface where you can:
1. Drag and drop an image or click to select one
2. View the original image alongside the annotated result
3. See detected objects with bounding boxes in real-time

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
├── examples/               # Example usage scripts
│   ├── client_example.py   # Python client example
│   ├── security_monitor.py # Continuous monitoring example
│   └── test_api.sh         # Bash testing script
├── models/                 # Model files (auto-downloaded)
│   └── efficientdet_lite0.tflite
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── download_model.sh      # Script to manually download model
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
- Try manually downloading the model using `./download_model.sh`
- Or download directly from the URL and place in `models/` directory

### Poor detection quality
- Adjust `score_threshold` in the detector options
- Ensure images are well-lit and in focus
- Try different camera angles

## Example Scripts

The `examples/` directory contains ready-to-use scripts:

### 1. Python Client Example (`client_example.py`)
Simple Python client demonstrating API usage:
```bash
python examples/client_example.py path/to/image.jpg
```

### 2. Security Camera Monitor (`security_monitor.py`)
Continuous monitoring script for home security cameras:
```bash
python examples/security_monitor.py /path/to/camera/frame.jpg
```

This script:
- Continuously monitors camera frames
- Triggers alerts for specific object categories (person, car, etc.)
- Logs all detections
- Can be customized for notifications, recording, etc.

### 3. API Testing Script (`test_api.sh`)
Bash script to test all API endpoints:
```bash
./examples/test_api.sh test_image.jpg
```

## CI/CD and Docker Hub Integration

This repository is configured with automated Docker builds and publishing to Docker Hub using GitHub Actions.

### Automatic Docker Builds

Every push to the `main` or `master` branch and every version tag triggers an automatic Docker build and push to Docker Hub.

### Docker Hub Images

Pre-built Docker images are available on Docker Hub (replace `<your-dockerhub-username>` with the actual username):
- **Latest stable**: `<your-dockerhub-username>/ml-models:latest`
- **Version tags**: `<your-dockerhub-username>/ml-models:v1.0.0`
- **Branch builds**: `<your-dockerhub-username>/ml-models:main`

### Setting Up Docker Hub Publishing (For Maintainers)

To enable automated Docker Hub publishing, configure the following GitHub secrets in your repository settings:

1. Go to your repository on GitHub
2. Navigate to Settings → Secrets and variables → Actions
3. Add the following secrets:
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: Your Docker Hub access token or password

**Note**: For security, it's recommended to use a Docker Hub access token instead of your password. You can create one at https://hub.docker.com/settings/security

### Workflow Triggers

The Docker build workflow is triggered by:
- **Push to main/master branch**: Builds and pushes with `latest` tag
- **Version tags** (e.g., `v1.0.0`): Builds and pushes with version tags (`1.0.0`, `1.0`, `1`, and `latest`)
- **Pull requests**: Builds only (no push to Docker Hub)

### Manual Docker Build and Push

To manually build and push to Docker Hub:

```bash
# Replace <your-dockerhub-username> with your actual Docker Hub username
# Build the image
docker build -t <your-dockerhub-username>/ml-models:latest .

# Log in to Docker Hub
docker login

# Push the image
docker push <your-dockerhub-username>/ml-models:latest
```

## License

This project uses MediaPipe which is licensed under Apache License 2.0.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
