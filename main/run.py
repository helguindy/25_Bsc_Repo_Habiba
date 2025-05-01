import os
import cv2
import pandas as pd
from ultralytics import YOLO
import easyocr
import numpy as np
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent

def estimate_distance(bbox_height, real_height=0.6):  # real_height in meters
    # Focal length (can be calibrated for your camera)
    focal_length = 1000  # This is an example value, should be calibrated for your camera
    
    # Calculate distance using triangle similarity
    distance = (real_height * focal_length) / bbox_height
    return distance

def calculate_euclidean_distance(bbox, image_shape):
    # Calculate center of bounding box
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate center of image
    image_center_x = image_shape[1] / 2
    image_center_y = image_shape[0] / 2
    
    # Calculate Euclidean distance
    distance = np.sqrt((center_x - image_center_x)**2 + (center_y - image_center_y)**2)
    return distance

class SpeedDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'])
        
    def detect_objects(self, image):
        results = self.model.predict(image, conf=0.15)[0]
        detections = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detection_info = {
                'class': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'ocr_number': None,
                'distance': None,
                'euclidean_distance': None
            }
            
            # Calculate distances
            bbox_height = y2 - y1
            detection_info['distance'] = estimate_distance(bbox_height)
            detection_info['euclidean_distance'] = calculate_euclidean_distance((x1, y1, x2, y2), image.shape)
            
            # Process speed limit signs
            if "Speed Limit" in class_name or "Speed limit" in class_name:
                roi = image[y1:y2, x1:x2]
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    ocr_results = self.reader.readtext(thresh_roi)
                    
                    for _, text, _ in ocr_results:
                        numbers = ''.join(filter(str.isdigit, text))
                        if numbers:
                            detection_info['ocr_number'] = numbers
                            break
            
            detections.append(detection_info)
        
        return detections

def main():
    root_dir = get_project_root()
    test_folder = root_dir / "my test images"
    model_path = root_dir / "runs" / "detect" / "traffic_signs" / "weights" / "best.pt"
    output_folder = root_dir / "my test images results"
    output_csv = output_folder / "detections.csv"
    
    output_folder.mkdir(exist_ok=True)
    
    print("Initializing detector...")
    detector = SpeedDetector(model_path)
    
    results = []
    
    image_files = [f for f in os.listdir(test_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for img_name in sorted(image_files):
        print(f"\nProcessing {img_name}")
        
        img_path = test_folder / img_name
        image = cv2.imread(str(img_path))
        
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        detections = detector.detect_objects(image)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            
            # Create label with new format
            if class_name in ['lightg', 'lightr']:
                label1 = f"lights -{det['distance']:.1f}m {det['confidence']:.2f}"
                label2 = f"[{'green' if class_name == 'lightg' else 'red'}]"
            elif "Speed Limit" in class_name and det['ocr_number']:
                label1 = f"Speed Limit ({det['ocr_number']}) -{det['distance']:.1f}m {det['confidence']:.2f}"
                label2 = None
            else:
                label1 = f"{class_name} -{det['distance']:.1f}m {det['confidence']:.2f}"
                label2 = None
            
            # Use purple color for all boxes
            box_color = (255, 0, 255)  # Purple/magenta
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            
            # Add label with background
            (label1_w, label1_h), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            
            # Draw top label
            cv2.rectangle(image, (x1, y1-label1_h-10), (x1+label1_w, y1), box_color, -1)
            cv2.putText(image, label1, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            # Draw bottom label if it exists (for traffic lights)
            if label2:
                (label2_w, label2_h), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(image, (x1, y2), (x1+label2_w, y2+label2_h+10), box_color, -1)
                cv2.putText(image, label2, (x1, y2+label2_h), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            # Add to results for CSV
            results.append({
                'image_name': img_name,
                'class': 'lights' if class_name in ['lightg', 'lightr'] else class_name,
                'confidence': det['confidence'],
                'distance': det['distance'],
                'euclidean_distance': det['euclidean_distance'],
                'ocr_number': det['ocr_number']
            })
        
        # Save processed image
        output_path = output_folder / f"processed_{img_name}"
        cv2.imwrite(str(output_path), image)
    
    # Save results to CSV with all columns
    if results:
        df = pd.DataFrame(results)
        # Reorder columns for better readability
        df = df[['image_name', 'class', 'confidence', 'distance', 'euclidean_distance', 'ocr_number']]
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    print(f"Processed images saved to: {output_folder}")

if __name__ == "__main__":
    main()