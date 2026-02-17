import os

import cv2

from src.utils import parse_time_str


def _load_face_cascade() -> cv2.CascadeClassifier | None:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Warning: Could not load Haar cascade from {cascade_path}")
        return None
    return cascade


def detect_primary_face_x(video_path: str, time_offset: str) -> float:
    """
    Detects the primary face in the video at the given timestamp using OpenCV Haar Cascades.
    Returns the normalized X center (0.0 - 1.0) of the face.
    If no face is found, returns 0.5 (center).
    """
    try:
        face_cascade = _load_face_cascade()
        if face_cascade is None:
            return 0.5

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video for framing analysis: {video_path}")
            return 0.5

        try:
            seconds = parse_time_str(time_offset)
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            success, image = cap.read()
        finally:
            cap.release()

        if not success:
            print(f"Warning: Could not read frame at {time_offset} for framing.")
            return 0.5

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return 0.5

        primary_face = max(faces, key=lambda f: f[2] * f[3])
        x, _, w, _ = primary_face
        _, img_w = image.shape[:2]
        center_x_px = x + (w / 2)
        center_x_norm = center_x_px / img_w

        return max(0.0, min(1.0, center_x_norm))

    except Exception as exc:
        print(f"Error during face detection (OpenCV): {exc}")
        return 0.5
