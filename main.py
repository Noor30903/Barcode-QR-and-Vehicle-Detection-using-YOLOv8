import streamlit as st
from PIL import Image
from pyzbar.pyzbar import decode  # Import pyzbar for barcode decoding
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model for barcode detection
barcode_model = YOLO('barcode.pt')  # Ensure the model file is correct

# Function to perform YOLO-based barcode detection
def perform_yolo_inference(model, uploaded_image):
    image = Image.open(uploaded_image)
    results = model(image)
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

# Streamlit app
def main():
    st.title("Integrated Barcode Detection with YOLO and Pyzbar")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Run detection when button is pressed
        if st.button("Run Detection"):
            with st.spinner('Running inference...'):
                try:
                    # Perform YOLO barcode detection
                    barcode_results = perform_yolo_inference(barcode_model, uploaded_image)

                    # Convert uploaded image to OpenCV format
                    image = Image.open(uploaded_image)
                    image_np = np.array(image)

                    # Draw YOLO detection results
                    for result in barcode_results:
                        # Extract bounding boxes
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = box.conf[0]

                            # Crop detected region for barcode decoding
                            cropped_region = image_np[y1:y2, x1:x2]

                            # Decode barcodes in the cropped region
                            barcodes = decode_barcodes(cropped_region)

                            # Draw bounding box around the detected area
                            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            st.write(f"YOLO Detection Confidence: {confidence:.2f}")

                            # Display decoded barcode information
                            for barcode in barcodes:
                                barcode_data = barcode["data"]
                                barcode_type = barcode["type"]
                                st.write(f"Decoded Data: {barcode_data} ({barcode_type})")

                                # Draw the decoded text on the image
                                # cv2.putText(image_np, f"{barcode_data} ({barcode_type})",
                                #             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                cv2.putText(image_np, f"{barcode_data} ({barcode_type})",
                                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                    # Display the final image with detections
                    st.image(image_np, caption="Detection Results", use_column_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

