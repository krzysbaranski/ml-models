"""
Example Python client for the Object Detection API
Demonstrates how to send images for detection and process results
"""
import requests
import base64
import sys
import argparse


def detect_objects(image_path, api_url="http://localhost:8000/detect", 
                   return_image=True, image_format="jpeg"):
    """
    Send an image to the API for object detection
    
    Args:
        image_path: Path to the image file
        api_url: API endpoint URL
        return_image: Whether to return annotated image
        image_format: Format for returned image (jpeg/png)
    
    Returns:
        Dictionary with detection results
    """
    try:
        # Prepare the file and parameters
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'return_image': str(return_image).lower(),
                'image_format': image_format
            }
            
            # Send request
            response = requests.post(api_url, files=files, data=data)
            response.raise_for_status()
            
            return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def detect_faces(image_path, api_url="http://localhost:8000/detect_faces",
                 return_image=True, image_format="jpeg"):
    """
    Send an image to the API for face detection
    
    Args:
        image_path: Path to the image file
        api_url: API endpoint URL
        return_image: Whether to return annotated image
        image_format: Format for returned image (jpeg/png)
    
    Returns:
        Dictionary with detection results
    """
    try:
        # Prepare the file and parameters
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'return_image': str(return_image).lower(),
                'image_format': image_format
            }
            
            # Send request
            response = requests.post(api_url, files=files, data=data)
            response.raise_for_status()
            
            return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def save_annotated_image(result, output_path):
    """
    Save the annotated image from the API response
    
    Args:
        result: API response dictionary
        output_path: Path to save the image
    """
    if 'annotated_image' in result:
        image_data = base64.b64decode(result['annotated_image'])
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f"Annotated image saved to {output_path}")
    else:
        print("No annotated image in response")


def main():
    """Main function to demonstrate API usage"""
    parser = argparse.ArgumentParser(description='Client for Object Detection and Face Detection API')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--mode', choices=['objects', 'faces'], default='objects',
                        help='Detection mode: objects or faces (default: objects)')
    parser.add_argument('--output', default='annotated_output.jpg',
                        help='Output path for annotated image (default: annotated_output.jpg)')
    
    args = parser.parse_args()
    
    print(f"Sending {args.image_path} to API for {args.mode} detection...")
    
    # Call the appropriate API
    if args.mode == 'faces':
        result = detect_faces(args.image_path)
    else:
        result = detect_objects(args.image_path)
    
    if result:
        print(f"\nDetected {result['count']} {args.mode}:")
        
        for i, detection in enumerate(result['detections'], 1):
            if args.mode == 'faces':
                print(f"\n  Face {i}:")
                print(f"    Confidence: {detection['score']:.2%}")
                print(f"    Bounding Box:")
                print(f"      X: {detection['bbox']['x']}")
                print(f"      Y: {detection['bbox']['y']}")
                print(f"      Width: {detection['bbox']['width']}")
                print(f"      Height: {detection['bbox']['height']}")
                if 'keypoints' in detection:
                    print(f"    Keypoints: {len(detection['keypoints'])} detected")
            else:
                print(f"\n  Object {i}:")
                print(f"    Category: {detection['category']}")
                print(f"    Confidence: {detection['score']:.2%}")
                print(f"    Bounding Box:")
                print(f"      X: {detection['bbox']['x']}")
                print(f"      Y: {detection['bbox']['y']}")
                print(f"      Width: {detection['bbox']['width']}")
                print(f"      Height: {detection['bbox']['height']}")
        
        # Save annotated image
        save_annotated_image(result, args.output)
        print("\nDone!")
    else:
        print("Failed to get results from API")
        sys.exit(1)


if __name__ == "__main__":
    main()
