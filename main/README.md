# **🚗 EgyDrive Scene-Level Automatic Annotation Tool**

This project uses a custom-trained **YOLOv8** model to detect Egyptian road signs and objects, and **EasyOCR** to extract numerical values from speed limit signs.

---

## 📌 **Features**
- 🟢 Detects **Traffic Lights** (Red and Green only — no Yellow in Egypt 😄)
- 🚫 Detects **Stop Signs**
- ⛔ Detects **No Entry Signs**
- ⚠️ Detects **Speed Bumps** (the colored ones 😅)
- 🪧 Detects **Cones**
- 🚧 Detects **Speed Limit Signs**
- 🔢 Recognizes **actual speed values** from speed limit signs using **EasyOCR** for verification
- 🖼️ Outputs **annotated images** with bounding boxes
- 📄 Generates **detailed CSV reports**
- 📁 Supports **batch processing**

---

## 🧠 **Model Overview**
- **Detection Model:** YOLOv8 (Ultralytics)
- **OCR Engine:** EasyOCR
- **Dataset:** Custom-labeled Egyptian road signs
- **Detection Confidence Threshold:** 0.15 (default)
- **Output:** Bounding boxes, labels, OCR numbers, confidence scores and estimated distance

---

## ⚙️ **Requirements**
- 🐍 **Python 3.8+**
- ⚡ **CUDA-capable GPU** (optional but recommended)
- 📦 **Install dependencies from `requirements.txt`:**

```bash
ultralytics>=8.0.0  
easyocr>=1.7.0  
opencv-python>=4.8.0  
numpy>=1.24.0  
pandas>=2.0.0  
torch>=2.0.0  
torchvision>=0.15.0  
Pillow>=10.0.0  
PyYAML>=6.0.1
```

## 📊 **Results**

### 1. **Detection Results**
The model detects various traffic signs and speed limits. Below are some sample outputs:

- **Annotated Images:**
  - Blue bounding boxes around detected speed limit signs
  - Text overlays showing the detected speed value and confidence
  
  ![Annotated Image Example](path_to_annotated_image.png)  
  *Example of an annotated image with bounding boxes and speed values.*

### 2. **Graphs and Metrics**
During the validation process, several performance metrics are generated. Below are some graphs showing the model's performance.

#### **Precision-Recall Curve**

![Precision-Recall Curve](https://raw.githubusercontent.com/helguindy/25_Bsc_Repo_Habiba/main/model/detect/traffic_signs/PR_curve.png)
 
*Precision vs Recall curve showing the model's trade-off between precision and recall.*

#### **Loss Curve**

![Loss Curve](https://github.com/helguindy/25_Bsc_Repo_Habiba/raw/main/model/detect/traffic_signs/results.png)
*Training and validation loss curve over epochs.*

#### **Confusion Matrix**

![Confusion Matrix](https://github.com/helguindy/25_Bsc_Repo_Habiba/raw/main/model/detect/traffic_signs/confusion_matrix_normalized.png)
*Confusion matrix showing true positives, false positives, true negatives, and false negatives.*

### 3. **Validation Batch Results**
The model's performance on a validation batch is shown below:

*Sample validation batch results showing detected traffic signs, confidence scores*
![batch0](https://github.com/helguindy/25_Bsc_Repo_Habiba/raw/main/model/detect/traffic_signs/val_batch0_pred.jpg)
![batch1](https://github.com/helguindy/25_Bsc_Repo_Habiba/raw/main/model/detect/traffic_signs/val_batch1_pred.jpg)
![batch2](https://github.com/helguindy/25_Bsc_Repo_Habiba/raw/main/model/detect/traffic_signs/val_batch2_pred.jpg)





