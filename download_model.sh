#!/bin/bash
# Script to download the MediaPipe object detection model

MODEL_URL="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
MODEL_DIR="models"
MODEL_PATH="$MODEL_DIR/efficientdet_lite0.tflite"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    exit 0
fi

echo "Downloading model from $MODEL_URL..."

# Try wget first
if command -v wget &> /dev/null; then
    wget -O "$MODEL_PATH" "$MODEL_URL"
    if [ $? -eq 0 ] && [ -f "$MODEL_PATH" ]; then
        echo "Model downloaded successfully to $MODEL_PATH"
        exit 0
    fi
fi

# Try curl if wget failed
if command -v curl &> /dev/null; then
    curl -L -o "$MODEL_PATH" "$MODEL_URL"
    if [ $? -eq 0 ] && [ -f "$MODEL_PATH" ]; then
        echo "Model downloaded successfully to $MODEL_PATH"
        exit 0
    fi
fi

# If both failed, print instructions
echo "Failed to download model automatically."
echo "Please manually download from: $MODEL_URL"
echo "And save to: $MODEL_PATH"
exit 1
