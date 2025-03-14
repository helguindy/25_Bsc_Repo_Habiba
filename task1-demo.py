import cv2
import numpy as np

def simple_scene_annotation_cv(frame):
    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Daytime/Nighttime 
    brightness = np.mean(gray)
    daytime_label = "Daytime" if brightness > 70 else "Nighttime"

    # Traffic Detection 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    traffic_label = "Traffic" if len(faces) > 3 else "No Traffic"

    # Weather Context 
    blur_amount = cv2.Laplacian(gray, cv2.CV_64F).var()
    weather_label = "Clear" if blur_amount > 40 else "Foggy"

 
    cv2.putText(frame, f"Time: {daytime_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Traffic: {traffic_label}", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Weather: {weather_label}", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = simple_scene_annotation_cv(frame)
    cv2.imshow('Environmental Context', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()