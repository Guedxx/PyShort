import cv2
import mediapipe as mp
import numpy as np
from src.utils import parse_time_str

def detect_primary_face_x(video_path: str, time_offset: str) -> float:
    """
    Detects the primary face in the video at the given timestamp.
    Returns the normalized X center (0.0 - 1.0) of the face.
    If no face is found, returns 0.5 (center).
    """
    try:
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        
        # Use full-range model (model_selection=1) for better detection at various distances
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video for framing analysis: {video_path}")
                return 0.5

            # Convert timestamp to milliseconds
            seconds = parse_time_str(time_offset)
            timestamp_ms = seconds * 1000
            
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            
            success, image = cap.read()
            if not success:
                print(f"Warning: Could not read frame at {time_offset} for framing.")
                cap.release()
                return 0.5

            # Convert BGR to RGB
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            
            cap.release()

            if not results.detections:
                # No face detected
                return 0.5
                
            # Find the largest face by area (width * height)
            # bounding_box is relative [0..1]
            primary_detection = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height)
            
            bbox = primary_detection.location_data.relative_bounding_box
            center_x = bbox.xmin + (bbox.width / 2)
            
            # Clamp between 0 and 1
            return max(0.0, min(1.0, center_x))
            
    except Exception as e:
        print(f"Error during face detection: {e}")
        return 0.5

