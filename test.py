import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import torch

# Initialize the models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
barcode_model = YOLO('barcode.pt')

# Define paths
images_folder = r"path/to/your/original_dataset/images"  # Update with actual path
annotation_file = r"path/to/your/original_dataset/annotations.tsv"  # Update with actual path
output_folder = "output_barcode_results"
os.makedirs(output_folder, exist_ok=True)

# Helper function to decode barcodes with ZBar
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

# Process a single image with YOLO and ZBar
def process_image(image_path, ground_truth_barcode, save_folder=None):
    """Process a single image using YOLO and Pyzbar, and save the result."""
    image = Image.open(image_path)
    image_np = np.array(image)

    # Perform YOLO-based barcode detection
    yolo_results = barcode_model(image_np)

    detected_barcodes = []
    yolo_correct = False  # Flag to track if YOLO detected the correct region

    # Process YOLO detections
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_region = image_np[y1:y2, x1:x2]

            # Decode barcodes within the detected area using ZBar
            barcodes = decode_barcodes(cropped_region)
            for barcode in barcodes:
                detected_barcodes.append(barcode["data"])

                # Check if the decoded barcode matches the ground truth
                if barcode["data"] == ground_truth_barcode:
                    yolo_correct = True  # Mark YOLO as successful for this region

                # Annotate the detected barcode
                cv2.putText(image_np, f"{barcode['data']} ({barcode['type']})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the processed image with annotations (optional)
    if save_folder:
        output_path = os.path.join(save_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, image_np)

    return detected_barcodes, yolo_correct

def evaluate_barcode_detection(images_folder, annotation_file, output_folder):
    """Evaluate the YOLO + Pyzbar barcode model on the original dataset with TSV annotations."""
    annotations_df = pd.read_csv(annotation_file, sep='\t')

    # Check if 'code' column exists in TSV
    if 'code' not in annotations_df.columns:
        print("Error: 'code' column not found in the annotation file.")
        return

    all_barcodes_pred = []
    all_barcodes_true = []
    yolo_correct_count = 0
    total_images = 0

    for idx, row in annotations_df.iterrows():
        image_path = os.path.join(images_folder, row['filename'])
        if not os.path.exists(image_path):
            print(f"Image {row['filename']} not found.")
            continue

        # Load ground truth barcode from the TSV file
        ground_truth_barcode = str(row['code'])
        all_barcodes_true.append(ground_truth_barcode)

        # Process the image and get detected barcodes and YOLO success flag
        detected_barcodes, yolo_correct = process_image(image_path, ground_truth_barcode, save_folder=output_folder)
        
        # Count YOLO success if it detected the correct barcode region
        if yolo_correct:
            yolo_correct_count += 1

        # Use the first detected barcode as the predicted result (if available)
        all_barcodes_pred.append(detected_barcodes[0] if detected_barcodes else None)
        total_images += 1

    # Filter out pairs where either prediction or ground truth is None
    filtered_true = []
    filtered_pred = []

    for true, pred in zip(all_barcodes_true, all_barcodes_pred):
        if true is not None and pred is not None:
            filtered_true.append(true)
            filtered_pred.append(pred)

    # Check if there's any valid data to evaluate
    if len(filtered_true) == 0 or len(filtered_pred) == 0:
        print("No valid barcode detections found for evaluation.")
        return

    # Calculate and print evaluation metrics for the YOLO+ZBar pipeline
    print("\n--- Barcode Detection Evaluation ---")
    print(f"YOLO Correct Region Detection Rate: {yolo_correct_count / total_images:.2f}")
    print(f"Accuracy: {accuracy_score(filtered_true, filtered_pred):.2f}")
    print(f"Precision: {precision_score(filtered_true, filtered_pred, average='weighted', zero_division=0):.2f}")
    print(f"Recall: {recall_score(filtered_true, filtered_pred, average='weighted', zero_division=0):.2f}")
    print(f"F1 Score: {f1_score(filtered_true, filtered_pred, average='weighted', zero_division=0):.2f}")
    print("\nEvaluation completed.")

if __name__ == "__main__":
    evaluate_barcode_detection(images_folder, annotation_file, output_folder)

