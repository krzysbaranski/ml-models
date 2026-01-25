"""
Example Python client for the Object Detection API
Demonstrates how to send images for detection and process results
"""
import requests
import base64
import sys


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
    if len(sys.argv) < 2:
        print("Usage: python client_example.py <image_path>")
        print("Example: python client_example.py my_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Sending {image_path} to API for detection...")
    
    # Call the API
    result = detect_objects(image_path)
    
    if result:
        print(f"\nDetected {result['count']} objects:")
        
        for i, detection in enumerate(result['detections'], 1):
            print(f"\n  Object {i}:")
            print(f"    Category: {detection['category']}")
            print(f"    Confidence: {detection['score']:.2%}")
            print(f"    Bounding Box:")
            print(f"      X: {detection['bbox']['x']}")
            print(f"      Y: {detection['bbox']['y']}")
            print(f"      Width: {detection['bbox']['width']}")
            print(f"      Height: {detection['bbox']['height']}")
        
        # Save annotated image
        save_annotated_image(result, 'annotated_output.jpg')
        print("\nDone!")
    else:
        print("Failed to get results from API")
        sys.exit(1)


if __name__ == "__main__":
    main()
