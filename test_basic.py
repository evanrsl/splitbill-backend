import pytest
import asyncio
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import numpy as np

from main import app
from services.ocr_service import OCRService
from services.image_processor import AdaptiveImageProcessor
from services.text_parser import AdaptiveReceiptParser
from models import OCRItem, ParseResult, ReceiptType

client = TestClient(app)


class TestBasicFunctionality:
    """Basic functionality tests"""

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data

    def test_extract_endpoint_no_file(self):
        """Test extract endpoint without file"""
        response = client.post("/api/v1/extract")
        assert response.status_code == 422  # Validation error

    def test_extract_endpoint_invalid_file_type(self):
        """Test extract endpoint with invalid file type"""
        response = client.post(
            "/api/v1/extract", files={"image": ("test.txt", b"hello", "text/plain")}
        )
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file type" in data["error"]


class TestImageProcessor:
    """Test image processing functionality"""

    def setup_method(self):
        self.processor = AdaptiveImageProcessor()

    def create_test_image(self, width=200, height=100, text="TEST RECEIPT"):
        """Create a simple test image with text"""
        # Create white image
        image = Image.new("RGB", (width, height), color="white")

        # Convert to numpy for OpenCV
        img_array = np.array(image)

        # Convert to bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format="PNG")
        return img_buffer.getvalue()

    def test_image_preprocessing(self):
        """Test basic image preprocessing"""
        test_image_bytes = self.create_test_image()

        # Should not raise exception
        processed = self.processor.preprocess_image(test_image_bytes)

        # Should return numpy array
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 2  # Should be grayscale

    def test_image_compression(self):
        """Test image compression"""
        # Create large image
        large_image_bytes = self.create_test_image(2000, 2000)

        # Compress
        compressed = self.processor.compress_image(large_image_bytes, max_size=1000000)

        # Should be smaller
        assert len(compressed) <= len(large_image_bytes)


class TestTextParser:
    """Test text parsing functionality"""

    def setup_method(self):
        self.parser = AdaptiveReceiptParser()

    def test_simple_text_parsing(self):
        """Test parsing simple receipt text"""
        test_text = """
        Restaurant Name
        ===============
        Cheeseburger         12.99
        Fries                 4.50
        Soda                  2.25
        
        Total               19.74
        """

        result = self.parser.parse_receipt(test_text)

        assert isinstance(result, ParseResult)
        assert len(result.items) >= 2  # Should find at least some items
        assert all(isinstance(item, OCRItem) for item in result.items)
        assert all(item.price > 0 for item in result.items)

    def test_grocery_receipt_parsing(self):
        """Test parsing grocery-style receipt"""
        test_text = """
        GROCERY STORE
        
        Apples 2 x $3.00     6.00
        Bread                2.50
        Milk                 4.25
        
        Subtotal           12.75
        Tax                 1.02
        Total              13.77
        """

        result = self.parser.parse_receipt(test_text)

        assert isinstance(result, ParseResult)
        assert len(result.items) >= 2

        # Check for quantity detection
        apple_items = [
            item for item in result.items if "apple" in item.description.lower()
        ]
        if apple_items:
            assert apple_items[0].quantity is not None


class TestOCRService:
    """Test main OCR service"""

    def setup_method(self):
        self.service = OCRService()

    def create_receipt_image(self):
        """Create a synthetic receipt image"""
        # Create a white image
        img = Image.new("RGB", (400, 600), color="white")

        # You would normally use PIL.ImageDraw to add text
        # For this test, we'll just use a simple white image

        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        return img_buffer.getvalue()

    @pytest.mark.asyncio
    async def test_service_info(self):
        """Test service info retrieval"""
        info = self.service.get_service_info()

        assert "tesseract_version" in info
        assert "image_processor" in info
        assert "text_parser" in info

    # Note: Full OCR tests would require actual receipt images
    # In a real test suite, you'd include test images in a fixtures folder


class TestModels:
    """Test data models"""

    def test_ocr_item_creation(self):
        """Test OCRItem model"""
        item = OCRItem(description="Test Item", price=12.99, confidence=0.95)

        assert item.description == "Test Item"
        assert item.price == 12.99
        assert item.confidence == 0.95
        assert item.quantity is None

    def test_parse_result_creation(self):
        """Test ParseResult model"""
        items = [
            OCRItem(description="Item 1", price=10.00, confidence=0.9),
            OCRItem(description="Item 2", price=5.50, confidence=0.85),
        ]

        result = ParseResult(
            items=items,
            confidence=0.88,
            strategy_used="test_strategy",
            receipt_type=ReceiptType.RESTAURANT,
        )

        assert len(result.items) == 2
        assert result.confidence == 0.88
        assert result.strategy_used == "test_strategy"
        assert result.receipt_type == ReceiptType.RESTAURANT


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
