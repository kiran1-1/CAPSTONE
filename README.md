# 🚗 **Parking Spot Finder** 

This project is a **real-time parking spot detection system** that leverages cutting-edge computer vision techniques to identify and classify parking spots. It supports two approaches: **CNN-based detection** and **YOLOv8-based detection**, each tailored for specific scenarios.

---

## ✨ **Features**
- Real-time parking lot monitoring.
- Classification of parking spots into:
  - **🟢 Empty**
  - **🔴 Non-Empty**
  - **🔵 Disabled**
- Aggregated parking spot counts with dynamic updates.
- Interactive parking lot selection and information display.
- Two implementation approaches:
  1. **CNN-based Approach**: High accuracy for fixed-camera setups.
  2. **YOLOv8-based Approach**: Robust real-time detection for dynamic environments like drone footage.

---

## 🛠 **Technologies Used**
- **Frontend**: HTML, CSS, JavaScript, Flask for integration.
- **Backend**: Python, Flask, OpenCV, ResNet50 (CNN), YOLOv8.
- **Tools**:
  - 📄 **CVAT**: Dataset annotation.
  - 🔄 **Albumentations**: Data augmentation.
  - 📊 **PKLot Dataset**: Parking spot images.
  - 🔍 **OpenCV**: Mask generation and bounding box extraction.

---

## 💻 **Approach 1: CNN-Based Detection**
### **Workflow**
1. **Input**: Video feed from a fixed overhead camera.
2. **Mask Generation**: OpenCV generates a mask to isolate parking spots.
3. **Bounding Box Extraction**: Parking spots are identified and extracted as bounding boxes.
4. **ResNet50 Classification**:
   - Each bounding box is classified into `empty`, `non-empty`, or `disabled`.
5. **Optimizations**:
   - Frame-skipping (predictions every 100 frames).
   - State-change detection to minimize redundant predictions.
   - Predefined coordinates for `disabled` spots to handle dataset imbalance.
6. **Output**: Spot counts are dynamically updated in the frontend.

### **Strengths**
- High accuracy: Training accuracy (97.5%), Validation accuracy (97.9%).
- Best suited for fixed-camera setups.

### **Limitations**
- Computationally expensive for real-time scenarios.
- Requires a static camera position.

---

## 🚀 **Approach 2: YOLOv8-Based Detection**
### **Workflow**
1. **Input**: Video feed from static or dynamic sources (e.g., drones).
2. **Bounding Box Detection**: YOLOv8 detects parking spots and classifies them into `empty`, `non-empty`, or `disabled`.
3. **Data Annotation**: CVAT is used for creating labeled bounding boxes.
4. **Output**: Real-time parking spot counts are displayed in the frontend.

### **Strengths**
- Robust to camera movement and dynamic environments.
- Faster frame processing, ideal for real-time use.

### **Limitations**
- Slightly lower accuracy due to dataset imbalance.
- Requires more labeled data for optimal performance.

---

## 🛠️ **How to Run the Project**
### **Setup**
1. Clone the repository:
   ```bash
   git clone (https://github.com/kiran1-1/CAPSTONE.git)
