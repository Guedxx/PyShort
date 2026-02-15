import cv2
import os
import numpy as np
from src.utils import parse_time_str

def detect_primary_face_x(video_path: str, time_offset: str) -> float:
    """
    Detects the primary face in the video at the given timestamp using OpenCV Haar Cascades.
    Returns the normalized X center (0.0 - 1.0) of the face.
    If no face is found, returns 0.5 (center).
    """
    try:
        # Load Haar Cascade
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video for framing analysis: {video_path}")
            return 0.5

        # Convert timestamp to milliseconds
        seconds = parse_time_str(time_offset)
        timestamp_ms = seconds * 1000
        
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        
        success, image = cap.read()
        cap.release()

        if not success:
            print(f"Warning: Could not read frame at {time_offset} for framing.")
            return 0.5

        # Convert to Grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            # Fallback: try with lower threshold if needed, or just return center
            # Let's try profile face just in case? No, usually frontal is enough for shorts.
            return 0.5
            
        # Find the largest face by area (w * h)
        # faces is list of (x, y, w, h)
        primary_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = primary_face
        
        # Calculate center relative to image width
        img_h, img_w = image.shape[:2]
        center_x_px = x + (w / 2)
        
        # Normalize
        center_x_norm = center_x_px / img_w
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, center_x_norm))
            
    except Exception as e:
        print(f"Error during face detection (OpenCV): {e}")
        return 0.5

