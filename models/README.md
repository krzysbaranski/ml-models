# Model Files Directory

This directory contains the TensorFlow Lite models used by the API.

## Required Models

### 1. Object Detection Model
- **File**: `efficientdet_lite0.tflite`
- **URL**: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite
- **Size**: ~4.4 MB
- **Purpose**: Detect objects in images (COCO dataset categories)

### 2. Face Detection Model
- **File**: `blaze_face_short_range.tflite`
- **URL**: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
- **Size**: ~225 KB
- **Purpose**: Detect faces in images

### 3. Gesture Recognition Model
- **File**: `gesture_recognizer.task`
- **URL**: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
- **Size**: ~9.8 MB
- **Purpose**: Recognize hand gestures (👍, 👎, ✌️, ☝️, ✊, 👋, 🤟)

## Downloading Models

### Automatic Download (Recommended)
Run the download script from the repository root:
```bash
./download_model.sh
```

### Manual Download
If automatic download fails, you can manually download the models:

```bash
cd models

# Download object detection model
wget https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite

# Download face detection model
wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# Download gesture recognition model
wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

Or using curl:
```bash
cd models

# Download object detection model
curl -L -o efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite

# Download face detection model
curl -L -o blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

# Download gesture recognition model
curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

## Notes

- All models are from [Google's MediaPipe project](https://developers.google.com/mediapipe)
- Models are automatically included in Docker builds
- The API will fail to start if required model files are missing

## Important Notes

### Gesture Recognition Model

The `gesture_recognizer.task` model file is **not included in the repository** due to its size (~9.8 MB). You must download it separately:

```bash
cd models
wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

Or use the automated download script from the repository root:
```bash
./download_model.sh
```

The API will start without this model, but gesture recognition endpoints will return a 503 error until the model is downloaded.

### Model Licensing

All models are from Google's MediaPipe project and are subject to the [Apache License 2.0](https://github.com/google/mediapipe/blob/master/LICENSE).
