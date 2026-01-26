"""
Example: Continuous monitoring for home security camera
Simulates processing camera frames at regular intervals
"""
import requests
import time
import os
from datetime import datetime


class SecurityCameraMonitor:
    """Monitor for processing security camera frames"""
    
    def __init__(self, api_url="http://localhost:8000/detect", 
                 interval=2.0, alert_categories=None):
        """
        Initialize the monitor
        
        Args:
            api_url: API endpoint URL
            interval: Seconds between frame checks
            alert_categories: List of categories to trigger alerts (e.g., ['person', 'car'])
        """
        self.api_url = api_url
        self.interval = interval
        self.alert_categories = alert_categories or ['person']
        self.detection_log = []
    
    def process_frame(self, frame_path):
        """
        Process a single camera frame
        
        Args:
            frame_path: Path to the frame image
            
        Returns:
            Detection results or None if error
        """
        try:
            with open(frame_path, 'rb') as f:
                files = {'file': f}
                data = {'return_image': 'false'}  # Faster without image
                
                response = requests.post(self.api_url, files=files, data=data)
                response.raise_for_status()
                
                return response.json()
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def check_for_alerts(self, detections):
        """
        Check if any detections match alert criteria
        
        Args:
            detections: List of detection results
            
        Returns:
            List of alert-worthy detections
        """
        alerts = []
        
        for detection in detections:
            if detection['category'] in self.alert_categories:
                if detection['score'] >= 0.7:  # Confidence threshold
                    alerts.append(detection)
        
        return alerts
    
    def trigger_alert(self, alerts, timestamp):
        """
        Trigger an alert (customize for your needs)
        
        Args:
            alerts: List of alert-worthy detections
            timestamp: When the detection occurred
        """
        print(f"\n🚨 ALERT at {timestamp}")
        print("=" * 50)
        
        for alert in alerts:
            print(f"  Detected: {alert['category']}")
            print(f"  Confidence: {alert['score']:.2%}")
            print(f"  Location: ({alert['bbox']['x']}, {alert['bbox']['y']})")
        
        print("=" * 50)
        
        # Here you could:
        # - Send a notification (email, SMS, push notification)
        # - Save the frame for review
        # - Trigger recording
        # - Turn on lights
        # - etc.
    
    def log_detection(self, timestamp, detections, alerts):
        """
        Log detection results
        
        Args:
            timestamp: When the detection occurred
            detections: All detections
            alerts: Alert-worthy detections
        """
        self.detection_log.append({
            'timestamp': timestamp,
            'total_detections': len(detections),
            'alerts': len(alerts),
            'details': alerts
        })
    
    def monitor(self, frame_source):
        """
        Start monitoring camera frames
        
        Args:
            frame_source: Path to camera frame (updated by camera software)
        """
        print(f"Starting security camera monitor...")
        print(f"Checking every {self.interval} seconds")
        print(f"Alert categories: {', '.join(self.alert_categories)}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                if not os.path.exists(frame_source):
                    print(f"Waiting for frame at {frame_source}...")
                    time.sleep(self.interval)
                    continue
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Process the frame
                result = self.process_frame(frame_source)
                
                if result:
                    detections = result.get('detections', [])
                    
                    # Check for alerts
                    alerts = self.check_for_alerts(detections)
                    
                    # Log the detection
                    self.log_detection(timestamp, detections, alerts)
                    
                    # Print status
                    if alerts:
                        self.trigger_alert(alerts, timestamp)
                    elif detections:
                        print(f"[{timestamp}] {len(detections)} object(s) detected (no alerts)")
                    else:
                        print(f"[{timestamp}] No objects detected")
                
                # Wait before next check
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            print(f"Total detections logged: {len(self.detection_log)}")
            
            # Print summary
            total_alerts = sum(log['alerts'] for log in self.detection_log)
            print(f"Total alerts: {total_alerts}")


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python security_monitor.py <frame_path>")
        print("\nExample:")
        print("  python security_monitor.py /path/to/camera/latest_frame.jpg")
        print("\nThe script will continuously check this file for new frames.")
        sys.exit(1)
    
    frame_path = sys.argv[1]
    
    # Configure the monitor
    monitor = SecurityCameraMonitor(
        api_url="http://localhost:8000/detect",
        interval=2.0,  # Check every 2 seconds
        alert_categories=['person', 'car', 'dog', 'cat']  # What to alert on
    )
    
    # Start monitoring
    monitor.monitor(frame_path)


if __name__ == "__main__":
    main()
