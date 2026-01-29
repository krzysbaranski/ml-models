#!/bin/bash
# Script to download the MediaPipe models

MODEL_DIR="models"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Download object detection model
OBJ_MODEL_URL="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
OBJ_MODEL_PATH="$MODEL_DIR/efficientdet_lite0.tflite"

# Download face detection model
FACE_MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_MODEL_PATH="$MODEL_DIR/blaze_face_short_range.tflite"

# Download gesture recognizer model
GESTURE_MODEL_URL="https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
GESTURE_MODEL_PATH="$MODEL_DIR/gesture_recognizer.task"

download_model() {
    local url=$1
    local path=$2
    local name=$3
    
    # Check if model already exists
    if [ -f "$path" ]; then
        echo "$name model already exists at $path"
        return 0
    fi
    
    echo "Downloading $name model from $url..."
    
    # Try wget first
    if command -v wget &> /dev/null; then
        wget -O "$path" "$url"
        if [ $? -eq 0 ] && [ -f "$path" ]; then
            echo "$name model downloaded successfully to $path"
            return 0
        fi
    fi
    
    # Try curl if wget failed
    if command -v curl &> /dev/null; then
        curl -L -o "$path" "$url"
        if [ $? -eq 0 ] && [ -f "$path" ]; then
            echo "$name model downloaded successfully to $path"
            return 0
        fi
    fi
    
    # If both failed, print instructions
    echo "Failed to download $name model automatically."
    echo "Please manually download from: $url"
    echo "And save to: $path"
    return 1
}

# Download both models
download_model "$OBJ_MODEL_URL" "$OBJ_MODEL_PATH" "Object detection"
OBJ_RESULT=$?

download_model "$FACE_MODEL_URL" "$FACE_MODEL_PATH" "Face detection"
FACE_RESULT=$?

download_model "$GESTURE_MODEL_URL" "$GESTURE_MODEL_PATH" "Gesture recognition"
GESTURE_RESULT=$?

# Exit with error if any download failed
if [ $OBJ_RESULT -ne 0 ] || [ $FACE_RESULT -ne 0 ] || [ $GESTURE_RESULT -ne 0 ]; then
    exit 1
fi

echo "All models downloaded successfully!"
exit 0
