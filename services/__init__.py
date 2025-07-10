"""
Services package for split-bill-backend

This package contains the core business logic for OCR processing and text parsing.
"""

from .ocr_service import OCRService
from .image_processor import AdaptiveImageProcessor
from .text_parser import AdaptiveReceiptParser
from .exceptions import (
    OCRError,
    ImageProcessingError,
    ParsingError,
    TesseractError,
    ValidationError,
)

__all__ = [
    "OCRService",
    "AdaptiveImageProcessor",
    "AdaptiveReceiptParser",
    "OCRError",
    "ImageProcessingError",
    "ParsingError",
    "TesseractError",
    "ValidationError",
]
