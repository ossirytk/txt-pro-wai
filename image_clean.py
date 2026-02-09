import glob
import subprocess
from pathlib import Path

import click
import cv2
import numpy as np

ANGLE_NONE = 0
ANGLE_90 = 90
ANGLE_180 = 180
ANGLE_270 = 270
CONTOUR_POINT_COUNT = 4
DESKEW_ANGLE_NEGATIVE_LIMIT = -45
DESKEW_ANGLE_LIMIT = 45
EMPTY_SIZE = 0
FULL_ROTATION_DEGREES = 360
MIDPOINT_INTENSITY = 127
MIN_DESKEW_DIMENSION = 50
ZERO = 0
THRESHOLD_NONE = 0
THIN_RUN_MIN = 1
THIN_RUN_MAX = 3


# Simple debug logger for per-step image info
def _log_step(debug_log: str, name: str, img: np.ndarray) -> None:
    if not debug_log:
        return
    try:
        with Path(debug_log).open("a", encoding="utf-8") as debug_file:
            debug_file.write(f"{name}: {img.shape}\n")
    except OSError:
        # Never fail processing due to logging
        return


# pip install click opencv-python numpy


# ----------------------------
# Utility: gentle unsharp mask
# ----------------------------
def unsharp_mask(
    gray: np.ndarray,
    amount: float = 1.0,
    radius: float = 1.0,
    threshold: int = 0,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), radius)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    if threshold > THRESHOLD_NONE:
        low_contrast = np.abs(gray - blurred) < threshold
        sharpened[low_contrast] = gray[low_contrast]
    return sharpened


# ----------------------------
# Remove shadows / uneven illumination
# ----------------------------
def remove_shadows(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    background = background.astype(np.float32)
    gray_f = gray.astype(np.float32)

    # Normalize: bring background estimate to midrange ~127
    bg_mean = np.mean(background)
    scale = 127.0 / (bg_mean + 1e-6)

    # Apply gentle correction
    norm = gray_f * scale
    return np.clip(norm, 0, 255).astype(np.uint8)


# ----------------------------
# Equalize lighting (CLAHE)
# ----------------------------
def equalize_lighting(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    return clahe.apply(gray)


# ----------------------------
# Image rotation
# ----------------------------
def rotate_image(bgr: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by 90, 180, or 270 degrees. 0 = no rotation."""
    if angle == ANGLE_NONE or angle % FULL_ROTATION_DEGREES == ANGLE_NONE:
        return bgr

    # OpenCV rotateCode: 0=90°CW, 1=180°, 2=90°CCW
    if angle == ANGLE_90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == ANGLE_180:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if angle == ANGLE_270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


# ----------------------------
# Perspective correction ("scan" effect)
# ----------------------------
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def four_point_warp(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def correct_perspective(bgr: np.ndarray) -> np.ndarray:
    img = bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Minimum contour area: require at least 10% of image area
    img_area = gray.shape[0] * gray.shape[1]
    min_area = img_area * 0.1

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, closed=True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, closed=True)
        if len(approx) == CONTOUR_POINT_COUNT:
            return four_point_warp(img, approx.reshape(4, 2))

    return bgr


# ----------------------------
# Skew correction (deskew)
# ----------------------------
def deskew(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw  # text -> white

    coords = np.column_stack(np.where(bw > ZERO))
    if coords.size == EMPTY_SIZE:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(ANGLE_90 + angle) if angle < DESKEW_ANGLE_NEGATIVE_LIMIT else -angle

    # Skip deskew if angle is extreme (would create very thin output)
    if abs(angle) > DESKEW_ANGLE_LIMIT:
        return gray

    (h, w) = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # If deskewing made the image too thin, reject it
    if deskewed.shape[1] < MIN_DESKEW_DIMENSION or deskewed.shape[0] < MIN_DESKEW_DIMENSION:
        return gray

    return deskewed


# ----------------------------
# Thresholding: Otsu or Sauvola
# ----------------------------
def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def threshold_sauvola(
    gray: np.ndarray,
    window: int = 25,
    k: float = 0.2,
    r: int = 128,
) -> np.ndarray:
    if window % 2 == ZERO:
        window += 1

    gray_f = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(window, window), normalize=True)
    mean_sq = cv2.boxFilter(
        gray_f * gray_f,
        ddepth=-1,
        ksize=(window, window),
        normalize=True,
    )
    var = mean_sq - mean * mean
    var[var < ZERO] = ZERO
    std = np.sqrt(var)

    thresh = mean * (1 + k * ((std / r) - 1))
    return (gray_f > thresh).astype(np.uint8) * 255


# ----------------------------
# Full preprocessing pipeline
# ----------------------------
def preprocess_image(  # noqa: PLR0912, PLR0913, PLR0915
    input_path: str,
    upscale_factor: float = 2.0,
    threshold_method: str = "sauvola",
    use_perspective: bool = True,  # noqa: FBT002
    sauvola_window: int = 25,
    sauvola_k: float = 0.2,
    sharpen_amount: float = 0.8,
    sharpen_radius: float = 1.2,
    min_width: int = 100,
    min_height: int = 100,
    debug_log: str = "",
    rotate: int = 0,
) -> np.ndarray:
    bgr = cv2.imread(input_path)
    if bgr is None:
        msg = f"Could not read image: {input_path}"
        raise ValueError(msg)

    # 0) Rotate if needed
    if rotate != ANGLE_NONE:
        bgr = rotate_image(bgr, rotate)
        _log_step(debug_log, f"after_rotate_{rotate}", bgr)

    # 1) Correct perspective (optional)
    if use_perspective:
        bgr = correct_perspective(bgr)
    _log_step(debug_log, "after_perspective_bgr", bgr)

    # 2) Remove shadows
    gray = remove_shadows(bgr)
    _log_step(debug_log, "after_remove_shadows", gray)

    # 3) Equalize lighting
    gray = equalize_lighting(gray)
    _log_step(debug_log, "after_equalize", gray)

    # 4) Upscale 2x bicubic
    h, w = gray.shape[:2]
    gray = cv2.resize(
        gray,
        (int(w * upscale_factor), int(h * upscale_factor)),
        interpolation=cv2.INTER_CUBIC,
    )
    _log_step(debug_log, "after_upscale", gray)

    # 5) Gentle sharpening
    gray = unsharp_mask(gray, amount=sharpen_amount, radius=sharpen_radius, threshold=3)
    _log_step(debug_log, "after_sharpen", gray)

    # 6) Deskew
    gray = deskew(gray)
    _log_step(debug_log, "after_deskew", gray)

    # 7) Thresholding
    if threshold_method.lower() == "sauvola":
        th = threshold_sauvola(gray, window=sauvola_window, k=sauvola_k, r=128)
    else:
        th = threshold_otsu(gray)
    _log_step(debug_log, "after_threshold", th)

    # Prefer dark text on light background (simple heuristic)
    if np.mean(th) < MIDPOINT_INTENSITY:
        th = 255 - th
    _log_step(debug_log, "after_invert_check", th)

    # Detect if thresholding produced too many thin lines (likely artifact).
    # Count horizontal runs of 1-3 pixels to detect pathological cases.
    thin_line_count = 0
    for row in th:
        # Find connected components of white pixels
        in_run = False
        run_length = 0
        for pixel in row:
            if pixel > MIDPOINT_INTENSITY:  # white
                if not in_run:
                    in_run = True
                    run_length = 1
                else:
                    run_length += 1
            else:
                if in_run and THIN_RUN_MIN <= run_length <= THIN_RUN_MAX:
                    thin_line_count += 1
                in_run = False
                run_length = 0

    # If more than 25% of rows are thin artifacts, apply morphological cleanup
    thin_threshold = th.shape[0] * 0.25
    if thin_line_count > thin_threshold:
        _log_step(
            debug_log,
            f"artifact_detection: {thin_line_count} thin lines (>{thin_threshold}), applying dilate+erode",
            th,
        )

        if threshold_method.lower() == "sauvola":
            # For Sauvola, try Otsu first
            th = threshold_otsu(gray)
            if np.mean(th) < MIDPOINT_INTENSITY:
                th = 255 - th
            _log_step(debug_log, "after_fallback_otsu", th)

        # Apply morphological cleanup to thicken text (works for both Otsu fallback and native Otsu)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.dilate(th, kernel, iterations=1)
        th = cv2.erode(th, kernel, iterations=1)
        _log_step(debug_log, "after_morph_cleanup", th)

    # Final safety: ensure a minimum size for Tesseract. If the result is
    # too small, upscale it (better than failing silently in Tesseract).
    h2, w2 = th.shape[:2]
    _log_step(debug_log, f"final_check: min={min_width}x{min_height}, actual={w2}x{h2}", th)
    if w2 < min_width or h2 < min_height:
        scale_x = max(1.0, min_width / float(w2))
        scale_y = max(1.0, min_height / float(h2))
        scale = max(scale_x, scale_y)
        new_w = round(w2 * scale)
        new_h = round(h2 * scale)
        th = cv2.resize(th, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        click.echo(
            f"[WARN] Small preprocessed image upscaled to {new_w}x{new_h} for Tesseract.",
            err=True,
        )
        _log_step(debug_log, "after_upscale_for_tesseract", th)

    _log_step(debug_log, "RETURN_FINAL", th)
    return th


# ----------------------------
# OCR runner: produces .txt
# ----------------------------
def run_tesseract(
    preprocessed_img_path: str,
    output_base: str,
    lang: str,
    psm: int,
    oem: int,
) -> None:
    cmd = [
        "tesseract",
        preprocessed_img_path,
        output_base,
        "--oem",
        str(oem),
        "--psm",
        str(psm),
        "-l",
        lang,
    ]
    click.echo(f"[DEBUG] Running: {' '.join(cmd)}", err=True)
    subprocess.run(cmd, check=True)  # noqa: S603


# ----------------------------
# Click CLI
# ----------------------------
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_glob", type=str)
@click.option(
    "--out",
    "out_dir",
    default="out",
    show_default=True,
    help="Output directory for .txt and intermediate .pre.png files.",
)
@click.option("--lang", default="eng", show_default=True, help="Tesseract language(s), e.g. 'eng' or 'fin+eng'.")
@click.option(
    "--psm",
    default=6,
    show_default=True,
    type=click.IntRange(0, 13),
    help="Tesseract page segmentation mode (0..13).",
)
@click.option(
    "--oem",
    default=1,
    show_default=True,
    type=click.IntRange(0, 3),
    help="Tesseract OCR engine mode (0..3).",
)
@click.option(
    "--threshold",
    "threshold_method",
    type=click.Choice(["sauvola", "otsu"], case_sensitive=False),
    default="sauvola",
    show_default=True,
    help="Binarization method.",
)
@click.option(
    "--upscale",
    "upscale_factor",
    default=2.0,
    show_default=True,
    type=click.FloatRange(1.0, 6.0),
    help="Upscale factor (bicubic). 2.0 is a good default for small fonts.",
)
@click.option(
    "--no-perspective",
    is_flag=True,
    help="Disable perspective correction (skip document contour detection).",
)
@click.option(
    "--rotate",
    default=0,
    show_default=True,
    type=click.Choice(["0", "90", "180", "270"]),
    help="Rotate image clockwise by angle (degrees) before processing.",
)
@click.option(
    "--sauvola-window",
    default=25,
    show_default=True,
    type=click.IntRange(9, 101),
    help="Sauvola window size (odd number recommended).",
)
@click.option(
    "--sauvola-k",
    default=0.2,
    show_default=True,
    type=click.FloatRange(0.05, 0.8),
    help="Sauvola k parameter.",
)
@click.option(
    "--sharpen-amount",
    default=0.8,
    show_default=True,
    type=click.FloatRange(0.0, 2.0),
    help="Unsharp mask amount (gentle sharpening).",
)
@click.option(
    "--sharpen-radius",
    default=1.2,
    show_default=True,
    type=click.FloatRange(0.3, 5.0),
    help="Unsharp mask radius.",
)
@click.option(
    "--keep-pre",
    is_flag=True,
    help="Keep preprocessed images (.pre.png). Default is to keep them anyway; use for clarity.",
)
@click.option(
    "--min-width",
    default=100,
    show_default=True,
    type=click.IntRange(10, 2000),
    help="Minimum width (px) for preprocessed image before sending to Tesseract.",
)
@click.option(
    "--min-height",
    default=100,
    show_default=True,
    type=click.IntRange(10, 2000),
    help="Minimum height (px) for preprocessed image before sending to Tesseract.",
)
@click.option(
    "--debug-log",
    default="",
    show_default=True,
    help="Path to append per-step debug info (image shapes).",
)
def main(  # noqa: PLR0913
    input_glob: str,
    out_dir: str,
    lang: str,
    psm: int,
    oem: int,
    threshold_method: str,
    upscale_factor: float,
    no_perspective: bool,
    rotate: str,
    sauvola_window: int,
    sauvola_k: float,
    sharpen_amount: float,
    sharpen_radius: float,
    keep_pre: bool,
    min_width: int,
    min_height: int,
    debug_log: str,
) -> None:
    """
    Phone-photo OCR pipeline -> plain .txt files using Tesseract.

    INPUT_GLOB example: "photos/*.jpg"
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    imgs = sorted(glob.glob(input_glob))  # noqa: PTH207
    if not imgs:
        msg = f"No images found for pattern: {input_glob}"
        raise click.ClickException(msg)

    use_perspective = not no_perspective

    for img_path in imgs:
        base = Path(img_path).stem
        pre_path = Path(out_dir) / f"{base}.pre.png"
        out_base = Path(out_dir) / base

        try:
            th = preprocess_image(
                img_path,
                upscale_factor=upscale_factor,
                threshold_method=threshold_method,
                use_perspective=use_perspective,
                sauvola_window=sauvola_window,
                sauvola_k=sauvola_k,
                sharpen_amount=sharpen_amount,
                sharpen_radius=sharpen_radius,
                min_width=min_width,
                min_height=min_height,
                debug_log=debug_log,
                rotate=int(rotate),
            )
            # Ensure image is uint8 before saving
            if th.dtype != np.uint8:
                th = th.astype(np.uint8)

            # Save using OpenCV with PNG compression
            success = cv2.imwrite(str(pre_path), th, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if debug_log:
                debug_path = Path(debug_log)
                with debug_path.open("a", encoding="utf-8") as debug_file:
                    debug_file.write(f"[main] returned image shape: {th.shape}\n")
                    debug_file.write(f"[main] cv2.imwrite returned: {success}\n")

                    if pre_path.exists():
                        file_size = pre_path.stat().st_size
                        verify_img = cv2.imread(str(pre_path), cv2.IMREAD_GRAYSCALE)
                        debug_file.write(f"[main] saved to {pre_path}, size={file_size} bytes\n")
                        verify_shape = verify_img.shape if verify_img is not None else "FAILED"
                        debug_file.write(f"[main] verified read: shape={verify_shape}\n")

                        # Extra validation can be added here if needed.

            run_tesseract(str(pre_path), str(out_base), lang=lang, psm=psm, oem=oem)

            click.echo(f"[OK] {img_path} -> {out_base}.txt")
            if not keep_pre:
                # If you actually want to auto-delete preprocessed files, uncomment:
                # os.remove(pre_path)
                pass
        except subprocess.CalledProcessError as e:
            click.echo(f"[TESSERACT FAIL] {img_path}: {e}", err=True)
        except Exception as e:
            click.echo(f"[FAIL] {img_path}: {e}", err=True)


if __name__ == "__main__":
    main()
