# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/

# Copy model file from repository
COPY models/efficientdet_lite0.tflite ./models/efficientdet_lite0.tflite
COPY models/blaze_face_short_range.tflite ./models/blaze_face_short_range.tflite
# Copy gesture recognizer model if it exists
COPY models/gesture_recognizer.task* ./models/ 2>/dev/null || true

# Expose port
EXPOSE 8000

# Set working directory to api folder
WORKDIR /app/api

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
