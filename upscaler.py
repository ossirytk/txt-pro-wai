import os
import argparse
import cv2
import time
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

MIN_REQUIRED_DISK_SPACE_GB = 50 # Minimum required disk space in GB

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k": (2560, 1440),
    "4k": (3840, 2160),
    "5k": (5120, 2880),
}

def get_input_resolution(input_path):
    """Gets the resolution of an image file."""
    input_ext = os.path.splitext(input_path)[1].lower()
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']

    if input_ext in image_extensions:
        img = cv2.imread(input_path)
        if img is not None:
            return img.shape[1], img.shape[0]
    return None

def initialize_upsampler(scale):
    """Initializes the Real-ESRGAN upsampler."""
    base_model_dir = os.path.join(os.path.dirname(__file__), "models")
    if scale == 4:
        model_path = os.path.join(base_model_dir, "RealESRGAN_x4plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif scale == 2:
        model_path = os.path.join(base_model_dir, "RealESRGAN_x2plus.pth")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        print("Error: Invalid scale factor. Please choose 2 or 4.")
        return None

    print(f"Initializing Real-ESRGAN model for {scale}x upscaling...")
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=0
    )
    return upsampler


def upscale_image(input_path, output_path, upsampler, target_resolution):
    """Upscales a single image and resizes to target resolution."""
    start_time = time.time()
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Cannot read image at {input_path}")
        return

    try:
        upscaled_img, _ = upsampler.enhance(img)
        final_img = cv2.resize(upscaled_img, target_resolution, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(output_path, final_img)
        end_time = time.time()
        print(f"Successfully upscaled image to {output_path}")
        print(f"Image processing took {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error upscaling image {input_path}: {e}")


def process_file(input_file, output_file, args, upsampler):
    input_resolution = get_input_resolution(input_file)
    if input_resolution is None:
        print(f"Skipping {input_file}: Could not determine input resolution.")
        return

    if args.target_resolution:
        if args.target_resolution in RESOLUTION_MAP:
            target_resolution = RESOLUTION_MAP[args.target_resolution]
        else:
            print("Error: Invalid target resolution.")
            return
    else:
        scale_factor = args.scale
        target_resolution = (int(input_resolution[0] * scale_factor), int(input_resolution[1] * scale_factor))

    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    
    input_ext = os.path.splitext(input_file)[1].lower()

    if input_ext in image_extensions:
        upscale_image(input_file, output_file, upsampler, target_resolution)
    else:
        print(f"Skipping {input_file}: Unsupported file format.")

def main():
    parser = argparse.ArgumentParser(description="Upscale an image using Real-ESRGAN.")
    parser.add_argument("input_path", help="Path to the input image file or directory.")
    parser.add_argument("output_path", help="Path to save the output file or directory.")
    parser.add_argument("--scale", type=float, help="Upscaling factor (e.g., 1.5, 2.0). Mutually exclusive with --target-resolution.")
    parser.add_argument("--target-resolution", type=str, help="Target resolution (e.g., 1080p, 4k). Mutually exclusive with --scale.")
    
    args = parser.parse_args()

    if args.scale and args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution, not both.")
        exit()

    if not args.scale and not args.target_resolution:
        print("Error: Please provide either --scale or --target-resolution.")
        exit()

    if args.scale:
        if args.scale <= 2:
            model_scale = 2
        elif args.scale <= 4:
            model_scale = 4
        else:
            print("Error: Scaling factors greater than 4 are not yet supported.")
            exit()
    else: # args.target_resolution
        if os.path.isdir(args.input_path):
            first_file = next((f for f in os.listdir(args.input_path) if os.path.isfile(os.path.join(args.input_path, f))), None)
            if not first_file:
                print("Error: Input directory is empty.")
                exit()
            input_resolution = get_input_resolution(os.path.join(args.input_path, first_file))
        else:
            input_resolution = get_input_resolution(args.input_path)
        
        if input_resolution is None:
            print("Error: Could not determine input resolution for the first file.")
            exit()
            
        if args.target_resolution in RESOLUTION_MAP:
            target_resolution = RESOLUTION_MAP[args.target_resolution]
            scale_factor = target_resolution[0] / input_resolution[0]
            if scale_factor <= 2:
                model_scale = 2
            elif scale_factor <= 4:
                model_scale = 4
            else:
                print("Error: Scaling factors greater than 4 are not yet supported.")
                exit()
        else:
            print("Error: Invalid target resolution.")
            exit()


    upsampler = initialize_upsampler(model_scale)
    if upsampler is None:
        exit()

    if os.path.isdir(args.input_path):
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        
        for filename in os.listdir(args.input_path):
            input_file = os.path.join(args.input_path, filename)
            output_file = os.path.join(args.output_path, filename)

            if os.path.isfile(input_file):
                process_file(input_file, output_file, args, upsampler)
    else:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        process_file(args.input_path, args.output_path, args, upsampler)


if __name__ == "__main__":
    main()
