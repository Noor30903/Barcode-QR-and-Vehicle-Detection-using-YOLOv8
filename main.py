import streamlit as st
from PIL import Image
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import torch

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the YOLO model for barcode detection
barcode_model = YOLO('barcode.pt')

# Initialize the EasyOCR reader with GPU support if available
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Function to perform YOLO-based barcode detection
def perform_yolo_inference(model, image_np):
    results = model(image_np)
    return results

# Function to decode barcodes using pyzbar
def decode_barcodes(image_np):
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

# Function to process a single uploaded image
def process_image(uploaded_image):
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Perform OCR on the uploaded image
    ocr_results = reader.readtext(image_np)

    # Perform YOLO-based barcode detection
    barcode_results = perform_yolo_inference(barcode_model, image_np)

    # Store results for displaying text below the image
    barcode_info = []
    ocr_info = []

    # Draw YOLO detection results, decode barcodes, and apply OCR results
    for result in barcode_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Crop detected region for barcode decoding
            cropped_region = image_np[y1:y2, x1:x2]

            # Decode barcodes in the cropped region
            barcodes = decode_barcodes(cropped_region)

            # Draw bounding box around the detected barcode area
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for barcode

            # Display decoded barcode information
            for barcode in barcodes:
                barcode_data = barcode["data"]
                barcode_type = barcode["type"]
                barcode_info.append(f"{barcode_data} ({barcode_type})")
                cv2.putText(image_np, f"{barcode_data} ({barcode_type})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Add OCR text to the image
    for (bbox, text, prob) in ocr_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw the OCR bounding box in red
        cv2.rectangle(image_np, top_left, bottom_right, (0, 0, 255), 2)  # Red for OCR

        # Display OCR text with a larger font size
        cv2.putText(image_np, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        ocr_info.append(f"Text: {text}")

    return image_np, barcode_info, ocr_info

# Streamlit app
def main():
    st.title("Integrated Barcode Detection with YOLO, Pyzbar, and EasyOCR")

    # Option to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            with st.spinner('Running inference...'):
                try:
                    # Process the uploaded image
                    result_image, barcode_info, ocr_info = process_image(uploaded_image)

                    # Display the final image with detections
                    st.image(result_image, caption="Detection Results", use_column_width=True)

                    # Display barcode information below the image
                    if barcode_info:
                        st.subheader("Detected Barcodes:")
                        for info in barcode_info:
                            st.write(info)
                    else:
                        st.write("No barcodes detected.")

                    # Display OCR information below the image
                    if ocr_info:
                        st.subheader("Detected Text:")
                        for info in ocr_info:
                            st.write(info)
                    else:
                        st.write("No text detected.")

                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
