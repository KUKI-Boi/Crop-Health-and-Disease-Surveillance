"""
Preprocessing Module.
Provides modular classical computer vision operations for crop image enhancement,
color space conversions (HSV, Lab, ExG), Gaussian blurring, and noise reduction.
"""

from typing import Tuple, Optional
import numpy as np
import cv2

def load_and_preprocess_image(
    image_bytes_or_path, 
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Loads an image from bytes or file path and converts to BGR format.

    Args:
        image_bytes_or_path: Bytes object or string file path.
        target_size (Optional[Tuple[int, int]]): Target (width, height) to resize if provided.

    Returns:
        np.ndarray: Loaded BGR image array.
    """
    if isinstance(image_bytes_or_path, bytes):
        nparr = np.frombuffer(image_bytes_or_path, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_bytes_or_path, str):
        img = cv2.imread(image_bytes_or_path)
    else:
        img = image_bytes_or_path

    if img is None:
        raise ValueError("Could not read image data.")

    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    return img

def convert_color_spaces(bgr_img: np.ndarray):
    """
    Converts BGR image to RGB, HSV, Lab, and calculates Excess Green Index (ExG).

    Args:
        bgr_img (np.ndarray): BGR image array.

    Returns:
        dict: Dictionary containing 'rgb', 'hsv', 'lab', 'exg' arrays.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    
    # Calculate Excess Green Index: ExG = 2G - R - B
    b, g, r = cv2.split(bgr_img.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_normalized = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "rgb": rgb,
        "hsv": hsv,
        "lab": lab,
        "exg": exg_normalized
    }

def apply_noise_reduction(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applies Gaussian Blur for noise reduction prior to thresholding.

    Args:
        img (np.ndarray): Input image array.
        kernel_size (int): Gaussian kernel size (odd integer).

    Returns:
        np.ndarray: Smoothed image array.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
