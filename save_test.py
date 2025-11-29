import sys
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def prepare_image_for_text(image_path, output_path="processed_image.png"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    im = Image.open(image_path)

    # Extract EXIF data
    exif_data = im._getexif()
    # Check if EXIF data is available
    if exif_data is not None:
        for tag_id, value in exif_data.items():
            # Get the tag name
            tag_name = TAGS.get(tag_id, tag_id)
            print(f"{tag_name}: {value}")
    else:
        print("No EXIF data found.") 

    im.save(output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    input_image = sys.argv[1]
    prepare_image_for_text(input_image)
