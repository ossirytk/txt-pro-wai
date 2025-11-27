import sys
import os
from PIL import Image

def prepare_image_for_text(image_path, output_path="processed_image.tiff"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    im = Image.open(image_path)
    im.save(output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    input_image = sys.argv[1]
    prepare_image_for_text(input_image)
