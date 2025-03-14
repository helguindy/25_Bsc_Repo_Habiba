import cv2
import json
import numpy as np

data = []
drawing = False
start_x, start_y = -1, -1

# Ask user if they want 2D or 3D
mode = input("Do you want to annotate in 2D or 3D? (Enter '2D' or '3D'): ").strip().upper()
if mode not in ["2D", "3D"]:
    print("Invalid ")
    exit()

# Load image
image_path = "images/image.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image.")
    exit()

def draw_3d_box(img, x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    depth = width // 4  

    # Front face (normal 2D rectangle)
    front_top_left = (x1, y1)
    front_top_right = (x2, y1)
    front_bottom_left = (x1, y2)
    front_bottom_right = (x2, y2)

    # Back face (shifted diagonally to give depth)
    back_top_left = (x1 + depth, y1 - depth)
    back_top_right = (x2 + depth, y1 - depth)
    back_bottom_left = (x1 + depth, y2 - depth)
    back_bottom_right = (x2 + depth, y2 - depth)

    # Draw front and back faces
    cv2.rectangle(img, front_top_left, front_bottom_right, (0, 255, 0), 2)
    cv2.rectangle(img, back_top_left, back_bottom_right, (0, 255, 0), 2)

    # Connect front and back faces
    cv2.line(img, front_top_left, back_top_left, (0, 255, 0), 2)
    cv2.line(img, front_top_right, back_top_right, (0, 255, 0), 2)
    cv2.line(img, front_bottom_left, back_bottom_left, (0, 255, 0), 2)
    cv2.line(img, front_bottom_right, back_bottom_right, (0, 255, 0), 2)

    return depth

def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, drawing, data, mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        
        # Ensure correct bounding box coordinates
        x1, y1 = min(start_x, end_x), min(start_y, end_y)
        x2, y2 = max(start_x, end_x), max(start_y, end_y)

        # Calculate width, height, and centroid
        width = x2 - x1
        height = y2 - y1
        centroid_x = x1 + width // 2
        centroid_y = y1 + height // 2
        
        # Ask for label
        label = input("Enter label for bounding box: ").strip()

        if mode == "2D":
            annotation = {
                "image": image_path,
                "type": "2D",
                "bounding_box": [x1, y1, x2, y2],
                "width": width,
                "height": height,
                "centroid": [centroid_x, centroid_y],
                "label": label
            }
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            depth = draw_3d_box(image, x1, y1, x2, y2)
            annotation = {
                "image": image_path,
                "type": "3D",
                "bounding_box": [x1, y1, x2, y2],
                "width": width,
                "height": height,
                "depth": depth,
                "centroid": [centroid_x, centroid_y],
                "label": label
            }

        data.append(annotation)
        
        # Draw label
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Manual Annotation", image)

cv2.namedWindow("Manual Annotation")
cv2.setMouseCallback("Manual Annotation", draw_rectangle)

while True:
    cv2.imshow("Manual Annotation", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to quit
        break

cv2.destroyAllWindows()

# Save annotations
if data:
    with open("annotations.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
        print("Annotations saved in annotations.json")
