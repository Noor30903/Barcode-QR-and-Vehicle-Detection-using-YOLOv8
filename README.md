# Drone-Assisted Inventory Management System

This project explores the development of a drone-assisted inventory management system using state-of-the-art AI techniques for barcode detection, decoding, and optical character recognition (OCR). The system is designed to improve inventory tracking and management efficiency in warehouses by automating the detection and identification of products.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset Information](#dataset-information)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## Overview
This project integrates machine learning and drone technologies to create a scalable inventory management system. Key components include:
1. **Barcode Detection**: Using YOLOv8 for detecting barcodes and QR codes.
2. **Barcode Decoding**: Leveraging the Zbar library for decoding detected barcode regions.
3. **OCR for Text Recognition**: Comparing EasyOCR, Tesseract, and TrOCR for recognizing shelf and zone identifiers.
4. **User Interface Design**: Providing a seamless UI for monitoring and managing inventory tasks.

---

## Features
- High-performance barcode detection using YOLOv8.
- Accurate barcode decoding using Zbar.
- Comparative analysis of OCR models for text recognition.
- Detailed performance metrics and visualizations for all experiments.
- User-friendly interface for real-time inventory tracking and feedback.

---

## Technologies Used
- **Python**: Programming language for model training and evaluation.
- **YOLOv8**: State-of-the-art object detection model.
- **Zbar**: Barcode decoding library.
- **EasyOCR, Tesseract, TrOCR**: OCR models for text recognition.
- **Ultralytics**: Framework for YOLOv8 integration.
- **OpenCV**: Image processing and manipulation.
- **Matplotlib**: Data visualization for performance analysis.

---

## Dataset Information
- **Barcode Dataset**: [Kaggle - Barcode Detection Dataset](https://www.kaggle.com/datasets/kushagrapandya/barcode-detection)
- **Goods Barcode Dataset**: [Kaggle - Goods Barcode Dataset](https://www.kaggle.com/datasets/kniazandrew/ru-goods-barcodes)
- **Synthetic OCR Dataset**: Custom-generated dataset containing alphanumeric identifiers and zone labels for OCR evaluation.

---

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo-name.git
    cd your-repo-name
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the required datasets and place them in the `datasets` directory.

4. Ensure you have the necessary hardware and software (e.g., a GPU for YOLOv8 training).

---

## Usage
1. **Barcode Detection**:
   - Train YOLOv8 on the barcode dataset:
     ```bash
     !yolo task=detect mode=train model=barcode.pt name=eval1 data=archive/data.yaml
     ```
   - Validate YOLOv8 performance:
     ```bash
     !yolo task=detect mode=val model=barcode.pt name=eval1 data=archive/data.yaml
     ```

2. **Barcode Decoding**:
   - Run the Zbar decoder on YOLO-detected regions:
     ```python
     python zbartest.py
     ```

3. **OCR Evaluation**:
   - Compare OCR models:
     ```python
     python ocrtest.py
     ```

4. **View Results**:
   - Access visual outputs and performance metrics in the `results` directory.

---
## Results
### 1. Barcode Detection
- **YOLOv8 Performance**:
  - mAP@0.5: 0.90
  - Precision: 0.858
  - Recall: 0.907

### 2. Barcode Decoding
- **Zbar**:
  - Decoding Accuracy: 0.75
  - Detection Rate: 0.93

### 3. OCR Comparison
- **EasyOCR**:
  - Exact Match Accuracy: 56%
  - Character-Level Accuracy: 91%
- **Tesseract**:
  - Exact Match Accuracy: 61%
  - Character-Level Accuracy: 89%
- **TrOCR**:
  - Exact Match Accuracy: 10%
  - Character-Level Accuracy: 60%

---

## Future Work
- Expand training datasets for better generalization.
- Improve preprocessing techniques to enhance barcode decoding accuracy.
- Incorporate autonomous drone navigation and control algorithms.
- Conduct real-world testing in warehouse environments.

##Team Members:
1. Noor Alawlaqi - S21107270
2. Maha Almashharawi - S20106480
3. Mashael Alsalamah - S20206926


