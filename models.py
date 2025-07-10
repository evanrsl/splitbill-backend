from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class OCRItem(BaseModel):
    """Represents a single line item extracted from a receipt"""

    description: str = Field(..., description="Item description/name")
    price: float = Field(..., description="Item price")
    quantity: Optional[int] = Field(None, description="Item quantity if detected")
    confidence: float = Field(
        default=1.0, description="Confidence score for this extraction"
    )


class ReceiptType(str, Enum):
    """Types of receipts for specialized parsing"""

    RESTAURANT = "restaurant"
    GROCERY = "grocery"
    RETAIL = "retail"
    GENERIC = "generic"


class OCRData(BaseModel):
    """Structured OCR output with spatial information"""

    text: str
    confidence: float
    bbox: Dict[str, int]  # bounding box coordinates
    line_num: int
    block_num: int


class ParseResult(BaseModel):
    """Result from a parsing strategy"""

    items: List[OCRItem]
    confidence: float = Field(
        ..., description="Overall confidence in the parsing result"
    )
    strategy_used: str = Field(..., description="Which parsing strategy was successful")
    receipt_type: Optional[ReceiptType] = None


class APIResponse(BaseModel):
    """Successful API response"""

    items: List[OCRItem]
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the parsing process",
    )


class APIError(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")


class HealthCheck(BaseModel):
    """Health check response"""

    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: str
