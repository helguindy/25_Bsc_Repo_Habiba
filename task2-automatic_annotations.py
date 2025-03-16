import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

def get_user_choice():
    choice = input("Choose annotation type (2D/3D/Both): ").strip().lower()
    while choice not in ["2d", "3d", "both"]:
        print("Invalid choice. Please enter 2D, 3D, or Both.")
        choice = input("Choose annotation type (2D/3D/Both): ").strip().lower()
    return choice

def annotate_2d(image):
    model = YOLO("yolov8n.pt")  # Load YOLO model
    results = model(image)
    annotations = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            annotations.append({
                "type": "2D",
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image, annotations

def annotate_3d(image):
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to("cpu").eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        depth = midas(img_tensor).squeeze().numpy()
    
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
    return depth_colored, {"type": "3D", "depth_map": depth.tolist()}

# Main Execution
choice = get_user_choice()
image_path = "images/image1.png"  # Change this to your input image path
image = cv2.imread(image_path)
annotations = []

if choice in ["2d", "both"]:
    image, ann_2d = annotate_2d(image)
    annotations.extend(ann_2d)

if choice in ["3d", "both"]:
    depth_image, ann_3d = annotate_3d(image)
    cv2.imwrite("output_depth.jpg", depth_image)  # Save 3D depth image
    annotations.append(ann_3d)
    
cv2.imwrite("output_annotated.jpg", image)  # Save 2D annotated image
with open("annotations.json", "w") as f:
    json.dump(annotations, f, indent=4)

print("2D and 3D annotations saved! Check 'output_annotated.jpg' and 'output_depth.jpg'.")
