import os
import random
import string
from PIL import Image, ImageDraw, ImageFont
import json

# Directory setup
dataset_dir = "ocr_dataset"
images_dir = os.path.join(dataset_dir, "images")
os.makedirs(images_dir, exist_ok=True)

# Fonts setup (adjust for your environment)
default_font_path = "C:\\Windows\\Fonts\\arial.ttf"  # Update for Windows
if not os.path.exists(default_font_path):
    default_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # For Linux/Mac

# Check if the font exists
if not os.path.exists(default_font_path):
    raise FileNotFoundError("No valid font found. Please update the font path.")

# Generate text identifiers
def generate_identifier():
    """Generate realistic shelf and zone identifiers."""
    if random.random() > 0.5:
        # Generate shelf ID: Alphanumeric
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(6, 12)))
    else:
        # Generate zone label: "Zone" + Letter
        return "Zone" + random.choice(string.ascii_uppercase)

# Function to calculate the maximum font size
def calculate_max_font_size(text, img_size, font_path, margin=20):
    """Incrementally find the largest font size that fits the image."""
    img_width, img_height = img_size
    font_size = 10
    while True:
        font = ImageFont.truetype(font_path, font_size)
        bbox = ImageDraw.Draw(Image.new("RGB", img_size)).textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width + margin * 2 > img_width or text_height + margin * 2 > img_height:
            return font_size - 5  # Step back slightly to ensure it fits
        font_size += 5

# Generate a full dataset with 400 images
num_images = 400
annotations = {}

for i in range(1, num_images + 1):
    identifier = generate_identifier()

    # Create a blank image
    img_size = (800, 400)  # Fixed image size
    img = Image.new("RGB", img_size, color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)

    # Determine the largest font size
    max_font_size = calculate_max_font_size(identifier, img_size, default_font_path)
    font = ImageFont.truetype(default_font_path, max_font_size)

    # Calculate text position for perfect centering
    bbox = draw.textbbox((0, 0), identifier, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (img_size[0] - text_width) // 2
    text_y = (img_size[1] - text_height) // 2

    # Draw the text
    draw.text((text_x, text_y), identifier, font=font, fill=(0, 0, 0))  # Black text

    # Save the image
    image_path = os.path.join(images_dir, f"image_{i:03d}.png")
    img.save(image_path)

    # Save the annotation
    annotations[f"image_{i:03d}.png"] = identifier

# Save annotations to a JSON file
annotations_file = os.path.join(dataset_dir, "annotations.json")
with open(annotations_file, "w") as f:
    json.dump(annotations, f, indent=4)


print(f"Dataset with {num_images} images created in directory: {dataset_dir}")
