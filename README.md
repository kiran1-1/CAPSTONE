# 🚗 **Parking Spot Finder**

This project is a **real-time parking spot detection system** that leverages cutting-edge computer vision techniques to identify and classify parking spots. It supports two approaches: **CNN-based detection** and **YOLOv8-based detection**, each tailored for specific scenarios.

---

## **Overview and Aim of the Project**

Parking in urban areas poses significant challenges, including traffic congestion, driver frustration, and wasted resources. This project aims to address these issues by developing a **real-time parking spot detection system**. The system uses computer vision to classify parking spots into `empty`, `non-empty`, or `disabled`.

### **Goals:**
- Provide real-time information about parking spot availability.
- Compare the effectiveness of two approaches:
  1. **CNN-based detection** for fixed-camera setups.
  2. **YOLOv8-based detection** for dynamic environments like drone footage.
- Enhance accuracy and efficiency in parking lot management.

---

## 📊 **Dataset Information**

### **Datasets Used:**
1. **PKLot Dataset**:
   - Contains labeled parking lot images under various conditions (e.g., lighting, weather).
   - Classes: `empty`, `non-empty`, `disabled`.

2. **Additional Data:**
   - Images collected from the internet to enhance dataset diversity.
   - Disabled parking spots were underrepresented, so data augmentation techniques were applied.

### **Data Preprocessing:**
- **For CNN:**
  - Images were cropped into individual parking spots and categorized into folders (`empty`, `non-empty`, `disabled`).
  - Augmentation techniques like brightness adjustment, noise addition, and shadow simulation were used to balance the dataset.
- **For YOLOv8:**
  - Bounding boxes were annotated using CVAT.
  - Augmentation was applied to simulate real-world conditions.

---

## 🔬 **Results**

### **CNN-Based Detection:**
- **Training Accuracy:** 97.5%
- **Validation Accuracy:** 97.9%
- **Strengths:**
  - High accuracy for structured parking lots.
  - Effective for fixed-camera setups.
- **Limitations:**
  - Computationally expensive for real-time applications.
  - Requires static camera positioning.

### **YOLOv8-Based Detection:**
- **Mean Average Precision (mAP):** 91.4%
- **Strengths:**
  - Robust to camera movement and dynamic environments.
  - Faster frame processing, ideal for real-time scenarios.
- **Limitations:**
  - Requires a balanced dataset for optimal performance.
  - Slightly lower accuracy compared to CNN.

---

## 📈 **Visuals**

### **Real-Time Parking Spot Status Visualization**
1. Parking spots are categorized into:
   - **🟢 Empty**
   - **🔴 Non-Empty**
   - **🔵 Disabled**
2. Users can select a parking lot and view real-time counts of parking spot availability.

### **Example Graphs:**
1. **Accuracy Over Epochs:**
   - Line graph comparing CNN and YOLOv8 training and validation accuracy.
2. **Frame Processing Time:**
   - Bar chart showing the average time taken by CNN and YOLOv8 to process a single frame.

---

## 📂 **Project Structure**
```plaintext
Parking-Spot-Finder/
│
├── app.py                 # Flask backend
├── static/                # Static files for frontend
│   ├── css/               # CSS files
│   └── js/                # JavaScript files
├── templates/             # HTML templates for frontend
├── models/                # Pre-trained models (ResNet50, YOLOv8)
├── utils/                 # Utility scripts (mask generation, preprocessing)
├── dataset/               # Dataset and annotations
└── requirements.txt       # Python dependencies
