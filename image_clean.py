import cv2
import sys
import os

def prepare_image_for_text(image_path, output_path="processed_image.png"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image format or corrupted file.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise with Gaussian blur
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)

    # Optional: Resize to improve OCR accuracy
    scale_factor = 1.5
    resized = cv2.resize(
        denoised,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC
    )

    # Save processed image
    cv2.imwrite(output_path, resized)
    print(f"Processed image saved to {output_path}")

    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    input_image = sys.argv[1]
    prepare_image_for_text(input_image)
