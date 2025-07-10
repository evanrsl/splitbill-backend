"""
Custom exception classes for the OCR service
"""


class OCRError(Exception):
    """Base exception for OCR-related errors"""

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ImageProcessingError(OCRError):
    """Raised when image preprocessing fails"""

    pass


class ParsingError(OCRError):
    """Raised when text parsing fails"""

    pass


class TesseractError(OCRError):
    """Raised when Tesseract OCR engine fails"""

    pass


class ValidationError(OCRError):
    """Raised when extracted data fails validation"""

    pass
