#!/bin/bash
# Example bash script for testing the API with curl

API_URL="http://localhost:8000"
IMAGE_PATH="${1:-test_image.jpg}"

echo "Testing Object Detection API"
echo "=============================="
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s "${API_URL}/health" | jq .
echo ""
echo ""

# Test root endpoint
echo "2. Testing root endpoint..."
curl -s "${API_URL}/" | jq .
echo ""
echo ""

# Test detect endpoint (JSON response with image)
echo "3. Testing /detect endpoint (with annotated image)..."
curl -s -X POST "${API_URL}/detect" \
  -F "file=@${IMAGE_PATH}" \
  -F "return_image=true" \
  -F "image_format=jpeg" \
  | jq '.detections, .count'
echo ""
echo ""

# Test detect endpoint (JSON only, no image)
echo "4. Testing /detect endpoint (JSON only)..."
curl -s -X POST "${API_URL}/detect" \
  -F "file=@${IMAGE_PATH}" \
  -F "return_image=false" \
  | jq .
echo ""
echo ""

# Test detect/image endpoint (image only)
echo "5. Testing /detect/image endpoint (saving image)..."
curl -s -X POST "${API_URL}/detect/image" \
  -F "file=@${IMAGE_PATH}" \
  -F "image_format=jpeg" \
  --output annotated_output.jpg

if [ -f annotated_output.jpg ]; then
    echo "✓ Annotated image saved to annotated_output.jpg"
    ls -lh annotated_output.jpg
else
    echo "✗ Failed to save annotated image"
fi

echo ""
echo "Testing complete!"
