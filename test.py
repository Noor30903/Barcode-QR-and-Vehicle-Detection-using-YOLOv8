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


def analyze_confidence_thresholds(y_true, y_scores):
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        # Convert confidence scores to binary predictions based on the threshold
        y_pred = [1 if score >= threshold else 0 for score in y_scores]

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Plot Precision, Recall, and F1 Score vs. Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold')
    plt.legend()
    plt.grid()
    plt.show()


def process_image(image_path, save_folder=None):
    """Process a single image using YOLO and Pyzbar, save result, and return detections with confidence."""
    image = Image.open(image_path)
    image_np = np.array(image)

    # Step 1: Decode full image with Pyzbar
    full_image_barcodes = decode_barcodes(image_np)

    # Step 2: Perform YOLO-based barcode detection
    barcode_results = barcode_model(image_np)
    detected_barcodes = []  # To store detected barcode data
    confidences = []        # To store corresponding confidence scores

    # Step 3: Process YOLO detections
    for result in barcode_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])  # Extract confidence score
            confidences.append(confidence)  # Save confidence

            cropped_region = image_np[y1:y2, x1:x2]

            # Decode barcodes within the YOLO-detected area
            yolo_barcodes = decode_barcodes(cropped_region)
            for barcode in yolo_barcodes:
                detected_barcodes.append((barcode["data"], confidence))  # Save data and confidence
                # Draw bounding boxes and annotations
                cv2.putText(image_np, f"{barcode['data']} ({barcode['type']})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the processed image if save_folder is provided
    if save_folder:
        output_path = os.path.join(save_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image_np)

    return detected_barcodes, confidences  # Return both barcodes and confidences





def evaluate_and_visualize(images_folder, annotation_file, output_folder):
    """Evaluate barcode detection with explicit negatives and visualize confusion matrix."""
    annotations_df = pd.read_csv(annotation_file, sep='\t')

    if 'code' not in annotations_df.columns:
        logging.error("Error: 'code' column not found in the annotation file.")
        return

    y_true = []  # Ground truth (1 for positive, 0 for negative)
    y_pred = []  # Predictions (1 for detected, 0 for not detected)
    y_scores = []  # Confidence scores from detections

    for idx, row in annotations_df.iterrows():
        image_path = os.path.join(images_folder, row['filename'])
        if not os.path.exists(image_path):
            logging.warning(f"Image {row['filename']} not found.")
            continue

        # Process image and get detected barcodes and confidences
        detected_barcodes, confidences = process_image(image_path, save_folder=output_folder)
        detected_barcodes_set = set([barcode.strip().lower() for barcode, _ in detected_barcodes])
        true_barcodes = set()

        # Parse ground truth barcodes
        if pd.notna(row['code']) and str(row['code']).strip() != "":
            true_barcodes = set([barcode.strip().lower() for barcode in str(row['code']).split(',')])

        # Handle True Positives and False Negatives
        for barcode in true_barcodes:
            y_true.append(1)  # Ground truth: barcode present
            y_pred.append(1 if barcode in detected_barcodes_set else 0)  # Detected or not
            y_scores.append(max([conf for b, conf in detected_barcodes if b == barcode], default=0))

        # Handle False Positives
        for barcode, conf in detected_barcodes:
            if barcode not in true_barcodes:
                y_true.append(0)  # Ground truth: no barcode
                y_pred.append(1)  # Detected
                y_scores.append(conf)

        # Skip True Negative logic since dataset has no negatives
        # This ensures no TNs are recorded
        if not true_barcodes and not detected_barcodes_set:
            continue

    # Set a global threshold for binary classification
    threshold = 0.5
    y_pred_final = [1 if score >= threshold else 0 for score in y_scores]

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
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_final, average='binary', zero_division=0)

    # Display results
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Perform threshold analysis
    analyze_confidence_thresholds(y_true, y_scores)





if __name__ == "__main__":
    evaluate_and_visualize(images_folder, annotation_file, output_folder)