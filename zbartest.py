import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import logging
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
barcode_model = YOLO('barcode.pt')

images_folder = r"data\test_dataset\goods-barcodes\images"
annotation_file = r"data\test_dataset\goods-barcodes\annotation.tsv"  # Adjust path as needed
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
            "bbox": (x, y, x + w, y + h)
        })
    return decoded_info

def process_image(image_path, ground_truth_barcodes):
    """Detect barcodes with YOLO, decode using Zbar, and evaluate."""
    image = Image.open(image_path)
    image_np = np.array(image)

    # YOLO Detection
    yolo_results = barcode_model(image_np)
    detected_barcodes = []  # For storing Zbar results

    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image_np[y1:y2, x1:x2]  # Crop detected barcode region

            # Decode the cropped region with Zbar
            zbar_results = decode_barcodes(cropped)
            for zbar_result in zbar_results:
                detected_barcodes.append(zbar_result["data"])

    # Evaluate Zbar results
    correct_decodings = sum(1 for barcode in detected_barcodes if barcode in ground_truth_barcodes)
    total_detections = len(detected_barcodes)

    return {
        "correct": correct_decodings,
        "total": len(ground_truth_barcodes),
        "detected": total_detections
    }

def evaluate(images_folder, annotation_file, output_folder):
    """Evaluate Zbar decoding after YOLO-based cropping."""
    annotations_df = pd.read_csv(annotation_file, sep='\t')

    if 'code' not in annotations_df.columns:
        logging.error("Error: 'code' column not found in the annotation file.")
        return

    correct_total = 0
    ground_truth_total = 0
    detections_total = 0

    for idx, row in annotations_df.iterrows():
        image_path = os.path.join(images_folder, row['filename'])
        if not os.path.exists(image_path):
            logging.warning(f"Image {row['filename']} not found.")
            continue

        # Parse ground truth barcodes
        ground_truth_barcodes = str(row['code']).split(',') if pd.notna(row['code']) else []

        # Process the image and accumulate results
        result = process_image(image_path, ground_truth_barcodes)
        correct_total += result["correct"]
        ground_truth_total += result["total"]
        detections_total += result["detected"]

    # Compute evaluation metrics
    accuracy = correct_total / ground_truth_total if ground_truth_total > 0 else 0
    detection_rate = correct_total / detections_total if detections_total > 0 else 0

    # Display results
    print(f"Zbar Decoding Accuracy: {accuracy:.2f}")
    print(f"Zbar Detection Rate (True Positive Rate): {detection_rate:.2f}")

if __name__ == "__main__":
    evaluate(images_folder, annotation_file, output_folder)
