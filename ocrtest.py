import json
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import easyocr
import torch
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Load the COCO dataset
with open(r'data\test_dataset\test\_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Directory containing images
image_dir = r"data\test_dataset\test"  # Update to actual image directory
output_dir = "output_ocr_results"

os.makedirs(output_dir, exist_ok=True)

# Evaluation Metrics Storage
true_texts = []
predicted_texts = []

# Function to draw bounding boxes and save images
def draw_and_save(image_path, annotations, ocr_results, output_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Use a better font if available

    # Draw ground truth bounding boxes and text
    for ann in annotations:
        bbox = ann["bbox"]
        x, y, w, h = map(int, bbox)
        draw.rectangle([(x, y), (x+w, y+h)], outline="blue", width=3)
        draw.text((x, y - 10), "Ground Truth", fill="blue", font=font)

    # Draw OCR results
    for (bbox, text, prob) in ocr_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        draw.rectangle([top_left, bottom_right], outline="red", width=3)
        draw.text(top_left, f"{text} ({prob:.2f})", fill="red", font=font)

    # Save processed image
    image.save(output_path)

# Process Images
for image_info in coco_data["images"]:
    image_path = os.path.join(image_dir, image_info["file_name"])
    output_path = os.path.join(output_dir, image_info["file_name"])
    annotations = [
        ann for ann in coco_data["annotations"] if ann["image_id"] == image_info["id"]
    ]

    if not os.path.exists(image_path):
        print(f"Image {image_info['file_name']} not found. Skipping...")
        continue

    # Load the image
    image_np = np.array(Image.open(image_path))

    # Perform OCR
    ocr_results = reader.readtext(image_np)

    # Collect ground truth and predictions for evaluation
    for ann in annotations:
        true_texts.append("Ground Truth Text")  # Replace with actual ground truth text if available
    for (_, text, _) in ocr_results:
        predicted_texts.append(text)

    # Draw bounding boxes and save processed images
    draw_and_save(image_path, annotations, ocr_results, output_path)
    print(f"Processed and saved {image_info['file_name']}")

# Evaluate Metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    true_texts, predicted_texts, average="weighted"
)

# Display Metrics
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Visualize Metrics
metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1}
plt.bar(metrics.keys(), metrics.values())
plt.title("OCR Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.show()
