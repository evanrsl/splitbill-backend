import re
import logging
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import pytesseract
from pytesseract import Output
import numpy as np

from models import OCRItem, OCRData, ParseResult, ReceiptType
from services.exceptions import ParsingError

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Abstract base class for parsing strategies"""

    @abstractmethod
    def parse(self, text_data: str) -> ParseResult:
        """Parse text and return structured result"""
        pass

    @abstractmethod
    def get_confidence(self, items: List[OCRItem]) -> float:
        """Calculate confidence score for parsed items"""
        pass


class SpatialParser(BaseParser):
    """
    Parser that uses OCR spatial data and confidence scores
    """

    def __init__(self):
        self.strategy_name = "spatial_parser"

    def parse(self, ocr_data: Dict) -> ParseResult:
        """
        Parse using spatial information from detailed OCR output

        Args:
            ocr_data: Dictionary containing OCR results with spatial info

        Returns:
            ParseResult with extracted items
        """
        try:
            items = []

            # Group text by lines using spatial proximity
            lines = self._group_by_lines(ocr_data)

            # Identify potential price columns
            price_columns = self._identify_price_columns(lines)

            # Extract items from each line
            for line in lines:
                item = self._extract_item_from_line(line, price_columns)
                if item:
                    items.append(item)

            confidence = self.get_confidence(items)
            receipt_type = self._classify_receipt_type(ocr_data.get("text", ""))

            return ParseResult(
                items=items,
                confidence=confidence,
                strategy_used=self.strategy_name,
                receipt_type=receipt_type,
            )

        except Exception as e:
            logger.error(f"Spatial parsing failed: {str(e)}")
            raise ParsingError(f"Spatial parsing failed: {str(e)}")

    def _group_by_lines(self, ocr_data: Dict) -> List[List[OCRData]]:
        """Group OCR elements by visual lines"""
        if not ocr_data or "text" not in ocr_data:
            return []

        elements = []
        texts = ocr_data["text"]
        confidences = ocr_data["conf"]
        lefts = ocr_data["left"]
        tops = ocr_data["top"]
        widths = ocr_data["width"]
        heights = ocr_data["height"]

        # Create OCRData objects
        for i, text in enumerate(texts):
            if text.strip() and int(confidences[i]) > 30:  # Filter low confidence
                elements.append(
                    OCRData(
                        text=text.strip(),
                        confidence=float(confidences[i]) / 100.0,
                        bbox={
                            "left": int(lefts[i]),
                            "top": int(tops[i]),
                            "width": int(widths[i]),
                            "height": int(heights[i]),
                        },
                        line_num=0,  # Will be set below
                        block_num=0,
                    )
                )

        # Group by vertical proximity (same line)
        lines = []
        elements.sort(key=lambda x: x.bbox["top"])  # Sort by vertical position

        current_line = []
        last_top = -1
        line_threshold = 20  # Pixels tolerance for same line

        for element in elements:
            if last_top == -1 or abs(element.bbox["top"] - last_top) <= line_threshold:
                current_line.append(element)
                last_top = element.bbox["top"]
            else:
                if current_line:
                    # Sort line elements by horizontal position
                    current_line.sort(key=lambda x: x.bbox["left"])
                    lines.append(current_line)
                current_line = [element]
                last_top = element.bbox["top"]

        if current_line:
            current_line.sort(key=lambda x: x.bbox["left"])
            lines.append(current_line)

        return lines

    def _identify_price_columns(self, lines: List[List[OCRData]]) -> List[int]:
        """Identify columns that likely contain prices"""
        price_pattern = re.compile(r"^\$?\d+\.?\d*$|^\d+\.\d{2}$")
        column_scores = {}

        for line in lines:
            for i, element in enumerate(line):
                if price_pattern.match(element.text.replace("$", "").replace(",", "")):
                    column_scores[i] = column_scores.get(i, 0) + 1

        # Return columns with high price frequency
        price_columns = [col for col, score in column_scores.items() if score >= 2]
        return price_columns or [len(lines[0]) - 1] if lines and lines[0] else []

    def _extract_item_from_line(
        self, line: List[OCRData], price_columns: List[int]
    ) -> Optional[OCRItem]:
        """Extract an item from a line of OCR data"""
        if len(line) < 2:  # Need at least description and price
            return None

        # Look for price in identified price columns
        price = None
        price_idx = -1

        for col_idx in price_columns:
            if col_idx < len(line):
                price_candidate = self._extract_price(line[col_idx].text)
                if price_candidate is not None:
                    price = price_candidate
                    price_idx = col_idx
                    break

        # If no price found in expected columns, search entire line
        if price is None:
            for i, element in enumerate(line):
                price_candidate = self._extract_price(element.text)
                if price_candidate is not None:
                    price = price_candidate
                    price_idx = i
                    break

        if price is None or price <= 0:
            return None

        # Build description from non-price elements
        description_parts = []
        for i, element in enumerate(line):
            if i != price_idx and not self._is_price_like(element.text):
                description_parts.append(element.text)

        description = " ".join(description_parts).strip()
        if not description:
            return None

        # Calculate confidence based on OCR confidence and price pattern match
        avg_confidence = sum(elem.confidence for elem in line) / len(line)
        price_confidence = 1.0 if self._is_price_like(line[price_idx].text) else 0.7

        return OCRItem(
            description=description,
            price=price,
            confidence=avg_confidence * price_confidence,
        )

    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text"""
        # Clean text
        cleaned = re.sub(r"[^\d\.\$,]", "", text)

        # Price patterns
        patterns = [
            r"\$?(\d+\.\d{2})",  # $12.34 or 12.34
            r"\$?(\d+)",  # $12 or 12
            r"(\d+),(\d{3})",  # 1,234
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    if "," in cleaned:
                        # Handle comma as thousands separator
                        price_str = cleaned.replace("$", "").replace(",", "")
                    else:
                        price_str = match.group(1)

                    price = float(price_str)
                    if 0.01 <= price <= 10000:  # Reasonable price range
                        return price
                except ValueError:
                    continue

        return None

    def _is_price_like(self, text: str) -> bool:
        """Check if text looks like a price"""
        price_pattern = re.compile(r"^\$?\d+\.?\d*$|^\d+\.\d{2}$")
        return bool(price_pattern.match(text.replace(",", "")))

    def _classify_receipt_type(self, text: str) -> ReceiptType:
        """Classify receipt type based on content"""
        text_lower = text.lower()

        restaurant_keywords = [
            "table",
            "server",
            "tip",
            "gratuity",
            "dine",
            "restaurant",
        ]
        grocery_keywords = ["grocery", "supermarket", "organic", "produce", "aisle"]
        retail_keywords = ["retail", "store", "size", "color", "department"]

        restaurant_score = sum(1 for kw in restaurant_keywords if kw in text_lower)
        grocery_score = sum(1 for kw in grocery_keywords if kw in text_lower)
        retail_score = sum(1 for kw in retail_keywords if kw in text_lower)

        if restaurant_score > max(grocery_score, retail_score):
            return ReceiptType.RESTAURANT
        elif grocery_score > retail_score:
            return ReceiptType.GROCERY
        elif retail_score > 0:
            return ReceiptType.RETAIL
        else:
            return ReceiptType.GENERIC

    def get_confidence(self, items: List[OCRItem]) -> float:
        """Calculate overall confidence for spatial parsing result"""
        if not items:
            return 0.0

        # Base confidence on item confidence and count
        avg_confidence = sum(item.confidence for item in items) / len(items)
        count_bonus = min(0.2, len(items) * 0.05)  # Bonus for more items

        return min(1.0, avg_confidence + count_bonus)


class PatternParser(BaseParser):
    """
    Parser using multiple regex patterns for different receipt types
    """

    def __init__(self):
        self.strategy_name = "pattern_parser"
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> Dict[ReceiptType, List[re.Pattern]]:
        """Initialize regex patterns for different receipt types"""
        return {
            ReceiptType.RESTAURANT: [
                re.compile(r"^(.+?)\s+\$?(\d+\.?\d*)$", re.MULTILINE),
                re.compile(r"^(.+?)\s+(\d+\.?\d*)$", re.MULTILINE),
                re.compile(r"(.+?)\s+\$(\d+\.\d{2})", re.MULTILINE),
            ],
            ReceiptType.GROCERY: [
                re.compile(r"^(.+?)\s+(\d+)\s*x\s*\$?(\d+\.?\d*)$", re.MULTILINE),
                re.compile(r"^(.+?)\s+\$?(\d+\.?\d*)$", re.MULTILINE),
            ],
            ReceiptType.RETAIL: [
                re.compile(r"^(.+?)\s+\$?(\d+\.?\d*)$", re.MULTILINE),
                re.compile(r"(.+?)\s+(\d+\.\d{2})", re.MULTILINE),
            ],
            ReceiptType.GENERIC: [
                re.compile(r"^(.+?)\s+\$?(\d+\.?\d*)$", re.MULTILINE),
                re.compile(r"(.+?)\s+(\d+\.\d{2})", re.MULTILINE),
                re.compile(r"(.+?)\s+(\d+)", re.MULTILINE),
            ],
        }

    def parse(self, text: str) -> ParseResult:
        """Parse using pattern matching"""
        try:
            receipt_type = self._classify_receipt_type(text)
            items = self._extract_with_patterns(text, receipt_type)

            if not items:
                # Try generic patterns as fallback
                items = self._extract_with_patterns(text, ReceiptType.GENERIC)

            confidence = self.get_confidence(items)

            return ParseResult(
                items=items,
                confidence=confidence,
                strategy_used=self.strategy_name,
                receipt_type=receipt_type,
            )

        except Exception as e:
            logger.error(f"Pattern parsing failed: {str(e)}")
            raise ParsingError(f"Pattern parsing failed: {str(e)}")

    def _extract_with_patterns(
        self, text: str, receipt_type: ReceiptType
    ) -> List[OCRItem]:
        """Extract items using patterns for specific receipt type"""
        items = []
        patterns = self.patterns.get(receipt_type, self.patterns[ReceiptType.GENERIC])

        # Clean text
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        for line in lines:
            # Skip obvious non-item lines
            if self._should_skip_line(line):
                continue

            for pattern in patterns:
                matches = pattern.findall(line)
                for match in matches:
                    item = self._create_item_from_match(match)
                    if item:
                        items.append(item)
                        break  # Found match, move to next line

        return self._deduplicate_items(items)

    def _create_item_from_match(self, match: Tuple) -> Optional[OCRItem]:
        """Create OCRItem from regex match"""
        try:
            if len(match) == 2:
                description, price_str = match
                price = float(price_str.replace("$", "").replace(",", ""))

                if 0.01 <= price <= 10000 and len(description.strip()) > 1:
                    return OCRItem(
                        description=description.strip(),
                        price=price,
                        confidence=0.8,  # Default confidence for pattern matching
                    )
            elif len(match) == 3:
                # Grocery format with quantity
                description, quantity_str, price_str = match
                quantity = int(quantity_str)
                unit_price = float(price_str.replace("$", "").replace(",", ""))
                total_price = quantity * unit_price

                if 0.01 <= total_price <= 10000 and len(description.strip()) > 1:
                    return OCRItem(
                        description=f"{description.strip()} (x{quantity})",
                        price=total_price,
                        quantity=quantity,
                        confidence=0.85,  # Higher confidence for quantity matches
                    )
        except (ValueError, IndexError):
            pass

        return None

    def _should_skip_line(self, line: str) -> bool:
        """Check if line should be skipped (not an item)"""
        skip_patterns = [
            r"^total",
            r"^subtotal",
            r"^tax",
            r"^tip",
            r"^gratuity",
            r"^change",
            r"^cash",
            r"^credit",
            r"^debit",
            r"^card",
            r"^thank you",
            r"^receipt",
            r"^order",
            r"^table",
            r"^\d{1,2}\/\d{1,2}\/\d{2,4}",  # dates
            r"^\d{1,2}:\d{2}",  # times
            r"^[A-Z]{2,}$",  # ALL CAPS short words
        ]

        line_lower = line.lower().strip()
        return any(re.match(pattern, line_lower) for pattern in skip_patterns)

    def _classify_receipt_type(self, text: str) -> ReceiptType:
        """Classify receipt type for pattern selection"""
        text_lower = text.lower()

        # Check for specific indicators
        if any(kw in text_lower for kw in ["table", "server", "dine in", "take out"]):
            return ReceiptType.RESTAURANT
        elif any(kw in text_lower for kw in ["grocery", "supermarket", "organic"]):
            return ReceiptType.GROCERY
        elif re.search(r"\d+\s*x\s*\$", text):  # quantity pattern
            return ReceiptType.GROCERY
        else:
            return ReceiptType.GENERIC

    def _deduplicate_items(self, items: List[OCRItem]) -> List[OCRItem]:
        """Remove duplicate items"""
        seen = set()
        unique_items = []

        for item in items:
            # Create a key based on description and price
            key = (item.description.lower().strip(), item.price)
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items

    def get_confidence(self, items: List[OCRItem]) -> float:
        """Calculate confidence for pattern parsing"""
        if not items:
            return 0.0

        # Higher confidence if we found multiple items with reasonable prices
        base_confidence = 0.7
        item_bonus = min(0.2, len(items) * 0.05)
        price_validity = sum(1 for item in items if 0.50 <= item.price <= 100) / len(
            items
        )

        return min(1.0, base_confidence + item_bonus + (price_validity * 0.1))


class FallbackParser(BaseParser):
    """
    Simple fallback parser for when other strategies fail
    """

    def __init__(self):
        self.strategy_name = "fallback_parser"

    def parse(self, text: str) -> ParseResult:
        """Simple extraction as last resort"""
        try:
            items = []
            lines = text.split("\n")

            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue

                # Very simple pattern: anything with a price-like ending
                price_match = re.search(r"(\d+\.?\d*)$", line)
                if price_match:
                    try:
                        price = float(price_match.group(1))
                        if 0.10 <= price <= 1000:
                            description = line[: price_match.start()].strip()
                            if description:
                                items.append(
                                    OCRItem(
                                        description=description,
                                        price=price,
                                        confidence=0.5,  # Low confidence fallback
                                    )
                                )
                    except ValueError:
                        continue

            return ParseResult(
                items=items[:10],  # Limit to 10 items for fallback
                confidence=self.get_confidence(items),
                strategy_used=self.strategy_name,
                receipt_type=ReceiptType.GENERIC,
            )

        except Exception as e:
            logger.error(f"Fallback parsing failed: {str(e)}")
            return ParseResult(
                items=[], confidence=0.0, strategy_used=self.strategy_name
            )

    def get_confidence(self, items: List[OCRItem]) -> float:
        """Conservative confidence for fallback"""
        return 0.4 if items else 0.0


class AdaptiveReceiptParser:
    """
    Main parser that tries multiple strategies adaptively
    """

    def __init__(self):
        self.parsers = [SpatialParser(), PatternParser(), FallbackParser()]
        self.confidence_threshold = 0.6

    def parse_receipt(self, text: str, ocr_data: Optional[Dict] = None) -> ParseResult:
        """
        Parse receipt using adaptive strategy selection

        Args:
            text: Raw text from OCR
            ocr_data: Optional detailed OCR data with spatial info

        Returns:
            ParseResult with best extraction
        """
        results = []

        # Try spatial parser first if we have detailed OCR data
        if ocr_data:
            try:
                spatial_result = self.parsers[0].parse(ocr_data)
                if spatial_result.confidence >= self.confidence_threshold:
                    logger.info(
                        f"Spatial parser succeeded with confidence {spatial_result.confidence:.2f}"
                    )
                    return spatial_result
                results.append(spatial_result)
            except Exception as e:
                logger.warning(f"Spatial parser failed: {str(e)}")

        # Try pattern parser
        try:
            pattern_result = self.parsers[1].parse(text)
            if pattern_result.confidence >= self.confidence_threshold:
                logger.info(
                    f"Pattern parser succeeded with confidence {pattern_result.confidence:.2f}"
                )
                return pattern_result
            results.append(pattern_result)
        except Exception as e:
            logger.warning(f"Pattern parser failed: {str(e)}")

        # Try fallback parser
        try:
            fallback_result = self.parsers[2].parse(text)
            results.append(fallback_result)
        except Exception as e:
            logger.warning(f"Fallback parser failed: {str(e)}")

        # Return best result or empty result
        if results:
            best_result = max(results, key=lambda x: x.confidence)
            logger.info(
                f"Best result from {best_result.strategy_used} with confidence {best_result.confidence:.2f}"
            )
            return best_result
        else:
            logger.error("All parsing strategies failed")
            return ParseResult(
                items=[],
                confidence=0.0,
                strategy_used="none",
                receipt_type=ReceiptType.GENERIC,
            )
