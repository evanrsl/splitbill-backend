import asyncio
import logging
from typing import Dict, Optional
import pytesseract
from pytesseract import Output
import numpy as np
import cv2

import re
from models import ParseResult, OCRItem
from services.image_processor import AdaptiveImageProcessor
from services.text_parser import AdaptiveReceiptParser
from services.exceptions import OCRError, TesseractError, ValidationError

logger = logging.getLogger(__name__)


class OCRService:
    """
    Main OCR service that orchestrates image processing and text extraction
    """

    def __init__(self):
        self.image_processor = AdaptiveImageProcessor()
        self.text_parser = AdaptiveReceiptParser()

        # Tesseract configuration
        self.tesseract_config = {
            "default": "--oem 3 --psm 6",  # Default: Uniform block of text
            "single_column": "--oem 3 --psm 4",  # Single column of text
            "sparse": "--oem 3 --psm 11",  # Sparse text
            "single_line": "--oem 3 --psm 8",  # Single text line
        }

        # Test Tesseract availability
        self._test_tesseract()

    def _test_tesseract(self):
        """Test if Tesseract is available and working"""
        try:
            # Create a simple test image
            test_image = np.ones((50, 200), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

            # Try to extract text
            result = pytesseract.image_to_string(test_image, config="--psm 8")
            if "TEST" not in result.upper():
                logger.warning("Tesseract test extraction may not be working optimally")
            else:
                logger.info("Tesseract is working correctly")

        except Exception as e:
            logger.error(f"Tesseract is not available or not working: {str(e)}")
            raise OCRError(
                "OCR engine not available",
                f"Tesseract OCR engine is not properly installed or configured: {str(e)}",
            )

    async def process_image_to_items(self, image_bytes: bytes) -> ParseResult:
        """
        Main processing pipeline: image → OCR → parsing → structured data

        Args:
            image_bytes: Raw image bytes

        Returns:
            ParseResult with extracted items
        """
        try:
            logger.info("Starting OCR processing pipeline")

            # Step 1: Compress image if too large
            if len(image_bytes) > 5_000_000:  # 5MB
                logger.info("Compressing large image")
                image_bytes = self.image_processor.compress_image(image_bytes)

            # Step 2: Preprocess image
            logger.info("Preprocessing image")
            processed_image = self.image_processor.preprocess_image(image_bytes)

            # Step 3: Extract text with multiple OCR strategies
            logger.info("Extracting text with OCR")
            text, ocr_data = await self._extract_text_adaptive(processed_image)

            if not text.strip():
                raise OCRError(
                    "No text extracted",
                    "OCR engine could not extract any readable text from the image",
                )

            logger.info(f"Extracted {len(text)} characters of text")

            # Step 4: Parse text to structured data
            logger.info("Parsing extracted text")
            result = self.text_parser.parse_receipt(text, ocr_data)

            # Step 5: Validate result
            validated_result = self._validate_and_enhance_result(result)

            logger.info(f"Successfully extracted {len(validated_result.items)} items")
            return validated_result

        except (OCRError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OCR pipeline: {str(e)}", exc_info=True)
            raise OCRError(
                "OCR processing failed",
                f"An unexpected error occurred during OCR processing: {str(e)}",
            )

    async def _extract_text_adaptive(
        self, processed_image: np.ndarray
    ) -> tuple[str, Optional[Dict]]:
        """
        Extract text using multiple OCR configurations adaptively

        Args:
            processed_image: Preprocessed image

        Returns:
            Tuple of (extracted_text, detailed_ocr_data)
        """
        configs_to_try = [
            ("default", self.tesseract_config["default"]),
            ("single_column", self.tesseract_config["single_column"]),
            ("sparse", self.tesseract_config["sparse"]),
        ]

        best_result = ""
        best_confidence = 0
        best_ocr_data = None

        for config_name, config in configs_to_try:
            try:
                logger.info(f"Trying OCR configuration: {config_name}")

                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(
                    processed_image, config=config, output_type=Output.DICT
                )

                # Get simple text extraction
                text = pytesseract.image_to_string(processed_image, config=config)

                # Calculate average confidence
                confidences = [int(conf) for conf in ocr_data["conf"] if int(conf) > 0]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0
                )

                logger.info(
                    f"Configuration {config_name}: {len(text)} chars, confidence: {avg_confidence:.1f}"
                )

                # Keep best result
                if avg_confidence > best_confidence and len(text.strip()) > len(
                    best_result.strip()
                ):
                    best_result = text
                    best_confidence = avg_confidence
                    best_ocr_data = ocr_data

                # If we got good confidence, we can stop
                if avg_confidence > 70 and len(text.strip()) > 50:
                    logger.info(f"Good OCR result with {config_name}, stopping early")
                    break

            except Exception as e:
                logger.warning(f"OCR configuration {config_name} failed: {str(e)}")
                continue

        if not best_result.strip():
            raise TesseractError(
                "OCR extraction failed",
                "No readable text could be extracted from the image with any configuration",
            )

        logger.info(
            f"Best OCR result: {len(best_result)} characters with confidence {best_confidence:.1f}"
        )
        return best_result, best_ocr_data

    def _validate_and_enhance_result(self, result: ParseResult) -> ParseResult:
        """
        Validate and enhance the parsing result

        Args:
            result: Raw parsing result

        Returns:
            Validated and enhanced result
        """
        if not result.items:
            raise ValidationError(
                "No items extracted",
                "Could not identify any valid line items in the receipt",
            )

        # Filter out invalid items
        valid_items = []
        for item in result.items:
            if self._validate_item(item):
                # Enhance item description
                enhanced_item = self._enhance_item(item)
                valid_items.append(enhanced_item)

        if not valid_items:
            raise ValidationError(
                "No valid items found", "All extracted items failed validation checks"
            )

        # Sort items by confidence
        valid_items.sort(key=lambda x: x.confidence, reverse=True)

        # Limit to reasonable number of items
        max_items = 50
        if len(valid_items) > max_items:
            logger.warning(
                f"Limiting results to {max_items} items (found {len(valid_items)})"
            )
            valid_items = valid_items[:max_items]

        # Recalculate overall confidence
        if valid_items:
            avg_confidence = sum(item.confidence for item in valid_items) / len(
                valid_items
            )
            confidence_adjustment = min(1.0, avg_confidence + (len(valid_items) * 0.02))
        else:
            confidence_adjustment = 0.0

        return ParseResult(
            items=valid_items,
            confidence=min(1.0, confidence_adjustment),
            strategy_used=result.strategy_used,
            receipt_type=result.receipt_type,
        )

    def _validate_item(self, item: OCRItem) -> bool:
        """
        Validate a single extracted item

        Args:
            item: Item to validate

        Returns:
            True if item is valid
        """
        # Check price range
        if not (0.01 <= item.price <= 10000):
            logger.debug(f"Invalid price range: {item.price}")
            return False

        # Check description length and content
        if len(item.description.strip()) < 2:
            logger.debug(f"Description too short: '{item.description}'")
            return False

        # Check for obviously non-item descriptions
        invalid_patterns = [
            r"^total",
            r"^subtotal",
            r"^tax",
            r"^tip",
            r"^change",
            r"^cash",
            r"^credit",
            r"^debit",
            r"^thank",
            r"^receipt",
            r"^\d+$",  # Just numbers
            r"^[^a-zA-Z]*$",  # No letters
        ]

        desc_lower = item.description.lower().strip()
        for pattern in invalid_patterns:
            if re.match(pattern, desc_lower):
                logger.debug(f"Invalid description pattern: '{item.description}'")
                return False

        # Check confidence
        if item.confidence < 0.1:
            logger.debug(f"Confidence too low: {item.confidence}")
            return False

        return True

    def _enhance_item(self, item: OCRItem) -> OCRItem:
        """
        Enhance item description by cleaning and normalizing

        Args:
            item: Original item

        Returns:
            Enhanced item
        """
        # Clean description
        description = item.description.strip()

        # Remove common OCR artifacts
        description = re.sub(r"\s+", " ", description)  # Multiple spaces
        description = re.sub(
            r"[^\w\s\-\(\)\&\']", "", description
        )  # Keep only safe chars

        # Capitalize properly
        if description.islower() or description.isupper():
            description = description.title()

        # Remove leading/trailing special characters
        description = description.strip(" -()&")

        return OCRItem(
            description=description,
            price=round(item.price, 2),  # Round to 2 decimal places
            quantity=item.quantity,
            confidence=item.confidence,
        )

    async def extract_text_only(self, image_bytes: bytes) -> str:
        """
        Extract raw text without parsing (for debugging)

        Args:
            image_bytes: Raw image bytes

        Returns:
            Raw extracted text
        """
        try:
            processed_image = self.image_processor.preprocess_image(image_bytes)
            text, _ = await self._extract_text_adaptive(processed_image)
            return text
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            raise OCRError(f"Text extraction failed: {str(e)}")

    def get_service_info(self) -> Dict[str, str]:
        """Get information about the OCR service configuration"""
        try:
            tesseract_version = pytesseract.get_tesseract_version()
        except:
            tesseract_version = "Unknown"

        return {
            "tesseract_version": str(tesseract_version),
            "image_processor": "AdaptiveImageProcessor",
            "text_parser": "AdaptiveReceiptParser",
            "max_image_size": "10MB",
            "supported_formats": "JPEG, PNG, BMP, TIFF",
        }
