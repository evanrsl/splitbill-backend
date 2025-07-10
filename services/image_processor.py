import cv2
import numpy as np
from typing import Tuple
import logging
from io import BytesIO
from PIL import Image, ImageEnhance
from services.exceptions import ImageProcessingError

logger = logging.getLogger(__name__)


class AdaptiveImageProcessor:
    """
    Handles image preprocessing with adaptive techniques for better OCR accuracy
    """

    def __init__(self):
        self.max_width = 2000  # Maximum width for processing
        self.max_height = 2000  # Maximum height for processing

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Main preprocessing pipeline that adapts to image characteristics

        Args:
            image_bytes: Raw image bytes

        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert bytes to OpenCV image
            image = self._bytes_to_cv2(image_bytes)

            # Analyze image characteristics
            is_low_contrast = self._is_low_contrast(image)
            is_skewed = self._is_skewed(image)
            is_noisy = self._is_noisy(image)

            logger.info(
                f"Image analysis - Low contrast: {is_low_contrast}, "
                f"Skewed: {is_skewed}, Noisy: {is_noisy}"
            )

            # Apply adaptive preprocessing
            processed = image.copy()

            # 1. Resize if too large
            processed = self._resize_if_needed(processed)

            # 2. Noise reduction if needed
            if is_noisy:
                processed = self._reduce_noise(processed)

            # 3. Deskew if needed
            if is_skewed:
                processed = self._deskew_image(processed)

            # 4. Enhance contrast if needed
            if is_low_contrast:
                processed = self._enhance_contrast(processed)

            # 5. Convert to grayscale
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # 6. Apply adaptive thresholding
            processed = self._apply_adaptive_threshold(processed)

            # 7. Morphological operations for cleanup
            processed = self._morphological_cleanup(processed)

            return processed

        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ImageProcessingError(
                "Failed to preprocess image",
                f"Error during image preprocessing: {str(e)}",
            )

    def _bytes_to_cv2(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format"""
        try:
            # Use PIL for better format support
            pil_image = Image.open(BytesIO(image_bytes))

            # Convert to RGB if necessary
            if pil_image.mode in ("RGBA", "P"):
                pil_image = pil_image.convert("RGB")

            # Convert to numpy array
            image_array = np.array(pil_image)

            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            return image_array

        except Exception as e:
            raise ImageProcessingError(
                "Invalid image format", f"Could not decode image: {str(e)}"
            )

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it's too large while maintaining aspect ratio"""
        height, width = image.shape[:2]

        if width <= self.max_width and height <= self.max_height:
            return image

        # Calculate scaling factor
        scale_w = self.max_width / width
        scale_h = self.max_height / height
        scale = min(scale_w, scale_h)

        new_width = int(width * scale)
        new_height = int(height * scale)

        logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _is_low_contrast(self, image: np.ndarray) -> bool:
        """Check if image has low contrast"""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        return np.std(gray) < 50  # Threshold for low contrast

    def _is_skewed(self, image: np.ndarray) -> bool:
        """Check if image appears to be skewed"""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # Use Hough line detection to find dominant angles
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines[:10]:  # Check first 10 lines
                angle = np.degrees(theta)
                angles.append(angle)

            # Check if there's a consistent skew
            angles = np.array(angles)
            mean_angle = np.mean(angles)
            # Consider skewed if mean angle is significantly off from 0, 90, or 180
            return abs(mean_angle % 90) > 5 and abs(mean_angle % 90) < 85

        return False

    def _is_noisy(self, image: np.ndarray) -> bool:
        """Check if image is noisy using Laplacian variance"""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < 100  # Threshold for noisy images

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction"""
        logger.info("Applying noise reduction")
        return cv2.bilateralFilter(image, 9, 75, 75)

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew the image using Hough line detection"""
        logger.info("Applying deskewing")

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is not None:
            angles = []
            for rho, theta in lines:
                angles.append(theta)

            # Calculate median angle
            median_angle = np.median(angles)
            angle_deg = np.degrees(median_angle) - 90

            # Rotate image
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

            return cv2.warpAffine(
                image,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        return image

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        logger.info("Enhancing contrast")

        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE to grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def _apply_adaptive_threshold(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction"""
        # Try multiple thresholding methods and pick the best one
        methods = [
            cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            ),
            cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            ),
        ]

        # Simple heuristic: choose the one with more white pixels (better for text)
        best_method = max(methods, key=lambda x: np.sum(x == 255))

        return best_method

    def _morphological_cleanup(self, binary_image: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up the binary image"""
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

        # Fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def compress_image(self, image_bytes: bytes, max_size: int = 2_000_000) -> bytes:
        """
        Compress image if it's too large for processing

        Args:
            image_bytes: Original image bytes
            max_size: Maximum size in bytes

        Returns:
            Compressed image bytes
        """
        if len(image_bytes) <= max_size:
            return image_bytes

        try:
            pil_image = Image.open(BytesIO(image_bytes))

            # Calculate compression ratio
            ratio = max_size / len(image_bytes)
            scale = np.sqrt(ratio)  # Scale both dimensions

            new_size = (int(pil_image.width * scale), int(pil_image.height * scale))
            compressed = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # Save with reduced quality
            output = BytesIO()
            compressed.save(output, format="JPEG", quality=85, optimize=True)

            logger.info(
                f"Compressed image from {len(image_bytes)} to {len(output.getvalue())} bytes"
            )

            return output.getvalue()

        except Exception as e:
            logger.warning(f"Failed to compress image: {str(e)}")
            return image_bytes
