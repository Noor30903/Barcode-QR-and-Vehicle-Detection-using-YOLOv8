import os
import json
import logging
import matplotlib.pyplot as plt
import easyocr
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OCR models
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
easyocr_reader = easyocr.Reader(['en'], gpu=device == 'cuda')
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Paths
annotations_file = r"ocr_dataset\annotations.json"
images_folder = r"ocr_dataset\images"
output_folder = "ocr_comparison_results"
os.makedirs(output_folder, exist_ok=True)

def save_annotated_image(image_path, ocr_results, output_path):
    """Draw bounding boxes and OCR text on the image and save it."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Load a larger font size for better visibility
    font_path = r"C:\Windows\Fonts\arial.ttf"  # Adjust for your system
    font = ImageFont.truetype(font_path, size=20)

    for result in ocr_results:
        bbox, text = result.get('bbox', None), result['text']
        if bbox:
            # Convert EasyOCR bounding box format [[x1, y1], [x2, y2], ...] to (x1, y1, x2, y2)
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            bbox_rect = [x1, y1, x2, y2]

            # Draw bounding box
            draw.rectangle(bbox_rect, outline="red", width=2)
            # Draw larger text above the bounding box
            draw.text((x1, y1 - 25), text, fill="blue", font=font)
        else:
            # For models without bounding boxes, draw the text in the top-left corner
            draw.text((10, 10), f"Extracted: {text}", fill="blue", font=font)

    image.save(output_path)


def easyocr_ocr(image_path):
    """Perform OCR using EasyOCR."""
    results = easyocr_reader.readtext(image_path)
    return [{"bbox": result[0], "text": result[1]} for result in results]

def tesseract_ocr(image_path):
    """Perform OCR using Tesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image).strip()
    return [{"bbox": None, "text": text}]  # Tesseract does not provide bounding boxes

def trocr_ocr(image_path):
    """Perform OCR using TrOCR."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values)
    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return [{"bbox": None, "text": text}]  # TrOCR does not provide bounding boxes

def evaluate_ocr_with_visualization(annotations_file, images_folder, output_folder):
    """Evaluate OCR models, save annotated images, and compare their performance."""
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    models = {
        "EasyOCR": easyocr_ocr,
        "Tesseract": tesseract_ocr,
        "TrOCR": trocr_ocr
    }

    results = {model: {"exact_matches": 0, "character_matches": 0, "total_characters": 0} for model in models}

    for model_name in models.keys():
        os.makedirs(os.path.join(output_folder, model_name), exist_ok=True)

    for image_name, ground_truth_text in annotations.items():
        image_path = os.path.join(images_folder, image_name)
        if not os.path.exists(image_path):
            logging.warning(f"Image {image_name} not found.")
            continue

        for model_name, ocr_function in models.items():
            try:
                ocr_results = ocr_function(image_path)

                # Save annotated image
                output_path = os.path.join(output_folder, model_name, image_name)
                save_annotated_image(image_path, ocr_results, output_path)

                # Extract OCR text for evaluation
                detected_text = " ".join([result['text'] for result in ocr_results])
                
                # Exact Match Accuracy
                if detected_text == ground_truth_text:
                    results[model_name]["exact_matches"] += 1
                
                # Character-Level Accuracy
                total_chars = len(ground_truth_text)
                correct_chars = sum(1 for a, b in zip(detected_text, ground_truth_text) if a == b)
                results[model_name]["character_matches"] += correct_chars
                results[model_name]["total_characters"] += total_chars
            except Exception as e:
                logging.error(f"Error with {model_name} on {image_name}: {e}")

    # Calculate metrics
    metrics = {
        model_name: {
            "exact_match_accuracy": result["exact_matches"] / len(annotations) if len(annotations) > 0 else 0,
            "character_accuracy": result["character_matches"] / result["total_characters"] if result["total_characters"] > 0 else 0
        }
        for model_name, result in results.items()
    }

    # Plot results
    plot_ocr_comparison_with_values(metrics)
    return metrics

def plot_ocr_comparison_with_values(metrics):
    """Plot OCR performance comparison with actual values annotated."""
    models = metrics.keys()
    exact_match_acc = [metrics[model]["exact_match_accuracy"] for model in models]
    char_acc = [metrics[model]["character_accuracy"] for model in models]

    x = range(len(models))  # X-axis positions for each model

    # Plot Exact Match Accuracy
    plt.figure(figsize=(10, 6))
    bar1 = plt.bar(x, exact_match_acc, color='blue', alpha=0.7, label='Exact Match Accuracy')
    bar2 = plt.bar(x, char_acc, color='green', alpha=0.7, label='Character-Level Accuracy', bottom=exact_match_acc)

    # Annotate bars with values
    for i, bar in enumerate(bar1):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() / 2, f"{exact_match_acc[i]:.2f}", ha='center', va='center', color='white', fontsize=10)
    for i, bar in enumerate(bar2):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + exact_match_acc[i] / 2, f"{char_acc[i]:.2f}", ha='center', va='center', color='white', fontsize=10)

    # X-axis and labels
    plt.xticks(x, models)
    plt.xlabel("OCR Models")
    plt.ylabel("Accuracy")
    plt.title("OCR Model Comparison with Actual Values")
    plt.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    metrics = evaluate_ocr_with_visualization(annotations_file, images_folder, output_folder)
    print("Evaluation Results:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}:")
        print(f"  Exact Match Accuracy: {model_metrics['exact_match_accuracy']:.2f}")
        print(f"  Character-Level Accuracy: {model_metrics['character_accuracy']:.2f}")
