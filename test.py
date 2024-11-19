import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
barcode_model = YOLO('barcode.pt')

images_folder = r"data\test_dataset\goods-barcodes\images"
annotation_file = r"data\test_dataset\goods-barcodes\annotation.tsv"
output_folder = "output_barcode_results"
os.makedirs(output_folder, exist_ok=True)

def decode_barcodes(image_np):
    """Decode barcodes using Pyzbar."""
    barcodes = decode(image_np)
    decoded_info = []
    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type
        x, y, w, h = barcode.rect
        decoded_info.append({
            "data": barcode_data,
            "type": barcode_type,
            "bbox": (x, y, w, h)
        })
    return decoded_info

def process_image(image_path, save_folder=None):
    """Process a single image using YOLO and Pyzbar, save result, and return detections."""
    image = Image.open(image_path)
    image_np = np.array(image)

    # Step 1: Decode full image with Pyzbar
    full_image_barcodes = decode_barcodes(image_np)

    # Step 2: Perform YOLO-based barcode detection
    barcode_results = barcode_model(image_np)
    detected_barcodes = []

    # Step 3: Process YOLO detections
    for result in barcode_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_region = image_np[y1:y2, x1:x2]

            # Decode barcodes within the YOLO-detected area
            yolo_barcodes = decode_barcodes(cropped_region)
            for barcode in yolo_barcodes:
                detected_barcodes.append(barcode["data"])
                # Draw bounding boxes and annotations
                cv2.putText(image_np, f"{barcode['data']} ({barcode['type']})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Step 4: Combine detections from YOLO and full-image Pyzbar scan
    all_detected_barcodes = {barcode["data"] for barcode in full_image_barcodes}
    all_detected_barcodes.update(detected_barcodes)  # Include YOLO-based detections

    # Save the processed image if save_folder is provided
    if save_folder:
        output_path = os.path.join(save_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image_np)

    return list(all_detected_barcodes)

def plot_f1_vs_threshold(y_true, y_scores):
    """Plot F1 Score vs. Threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores[:-1], marker='.', label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Threshold')
    plt.legend()
    plt.grid()
    plt.show()

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_scores):
    """Plot ROC Curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def evaluate_and_visualize(images_folder, annotation_file, output_folder):
    """Evaluate barcode detection with explicit negatives, and visualize confusion matrix + ROC."""
    annotations_df = pd.read_csv(annotation_file, sep='\t')

    if 'code' not in annotations_df.columns:
        logging.error("Error: 'code' column not found in the annotation file.")
        return

    y_true = []  # Ground truth (1 for positive, 0 for negative)
    y_pred = []  # Predictions (1 for detected, 0 for not detected)

    for idx, row in annotations_df.iterrows():
        image_path = os.path.join(images_folder, row['filename'])
        if not os.path.exists(image_path):
            logging.warning(f"Image {row['filename']} not found.")
            continue

        # Ground truth and predictions
        detected_barcodes = set(process_image(image_path, save_folder=output_folder))
        true_barcodes = set(str(row['code']).split(',')) if pd.notna(row['code']) else set()

        # Handle True Positives and False Negatives
        for barcode in true_barcodes:
            y_true.append(1)  # Ground truth: barcode present
            y_pred.append(1 if barcode in detected_barcodes else 0)  # Prediction: detected or not

        # Handle False Positives
        for barcode in detected_barcodes:
            if barcode not in true_barcodes:
                y_true.append(0)  # Ground truth: no barcode
                y_pred.append(1)  # Prediction: detected

        # Handle True Negatives
        if not true_barcodes and not detected_barcodes:
            y_true.append(0)  # Ground truth: no barcode
            y_pred.append(0)  # Prediction: no barcode

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix for Barcode Detection")
    plt.show()

    # Calculate evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    plot_f1_vs_threshold(y_true, y_pred)
    plot_roc_curve(y_true, y_pred)
    # Display results
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")



if __name__ == "__main__":
    evaluate_and_visualize(images_folder, annotation_file, output_folder)