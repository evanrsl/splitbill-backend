from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from typing import Union
import asyncio
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import APIResponse, APIError, HealthCheck, OCRItem
from services.ocr_service import OCRService
from services.exceptions import OCRError, ImageProcessingError, ParsingError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Split Bill OCR Backend",
    description="Serverless OCR service for extracting bill items from receipt images",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize OCR service
ocr_service = OCRService()

# Supported file types
SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/tiff",
}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = datetime.now()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    # Log response
    process_time = datetime.now() - start_time
    logger.info(
        f"Response: {response.status_code} - Time: {process_time.total_seconds():.2f}s"
    )

    return response


@app.exception_handler(OCRError)
async def ocr_error_handler(request: Request, exc: OCRError):
    """Handle OCR-specific errors"""
    logger.error(f"OCR Error: {exc.message} - Details: {exc.details}")
    return JSONResponse(
        status_code=422,
        content=APIError(
            error=exc.message, details=exc.details, error_code="OCR_FAILED"
        ).dict(),
    )


@app.exception_handler(ImageProcessingError)
async def image_processing_error_handler(request: Request, exc: ImageProcessingError):
    """Handle image processing errors"""
    logger.error(f"Image Processing Error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content=APIError(
            error=exc.message, details=exc.details, error_code="IMAGE_PROCESSING_FAILED"
        ).dict(),
    )


@app.exception_handler(ParsingError)
async def parsing_error_handler(request: Request, exc: ParsingError):
    """Handle text parsing errors"""
    logger.error(f"Parsing Error: {exc.message}")
    return JSONResponse(
        status_code=422,
        content=APIError(
            error=exc.message, details=exc.details, error_code="PARSING_FAILED"
        ).dict(),
    )


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy", version="1.0.0", timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/extract", response_model=APIResponse)
async def extract_items(image: UploadFile = File(...)):
    """
    Extract line items from a receipt image

    Args:
        image: Receipt image file (PNG, JPG, JPEG, BMP, TIFF)

    Returns:
        APIResponse with extracted items and metadata

    Raises:
        HTTPException: For various error conditions
    """

    # Validate file type
    if image.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=APIError(
                error="Unsupported file type",
                details=f"Supported types: {', '.join(SUPPORTED_MIME_TYPES)}",
                error_code="INVALID_FILE_TYPE",
            ).dict(),
        )

    # Validate file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB

    try:
        # Read image bytes
        image_bytes = await image.read()

        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail=APIError(
                    error="File too large",
                    details=f"Maximum file size is {max_size // (1024*1024)}MB",
                    error_code="FILE_TOO_LARGE",
                ).dict(),
            )

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail=APIError(
                    error="Empty file",
                    details="The uploaded file is empty",
                    error_code="EMPTY_FILE",
                ).dict(),
            )

        logger.info(
            f"Processing image: {image.filename}, Size: {len(image_bytes)} bytes"
        )

        # Process the image
        parse_result = await ocr_service.process_image_to_items(image_bytes)

        logger.info(
            f"Successfully extracted {len(parse_result.items)} items using {parse_result.strategy_used}"
        )

        # Prepare response
        return APIResponse(
            items=parse_result.items,
            metadata={
                "strategy_used": parse_result.strategy_used,
                "confidence": parse_result.confidence,
                "receipt_type": (
                    parse_result.receipt_type.value
                    if parse_result.receipt_type
                    else None
                ),
                "item_count": len(parse_result.items),
                "filename": image.filename,
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=APIError(
                error="Internal server error",
                details="An unexpected error occurred while processing the image",
                error_code="INTERNAL_ERROR",
            ).dict(),
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Split Bill OCR Backend",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "extract": "/api/v1/extract",
            "docs": "/docs",
        },
    }


# For local development
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
