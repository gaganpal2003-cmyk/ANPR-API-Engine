from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int = Field(..., description="Top-left X coordinate")
    y1: int = Field(..., description="Top-left Y coordinate")
    x2: int = Field(..., description="Bottom-right X coordinate")
    y2: int = Field(..., description="Bottom-right Y coordinate")
    width: Optional[int] = Field(None, description="Box width in pixels")
    height: Optional[int] = Field(None, description="Box height in pixels")


class CharacterPrediction(BaseModel):
    char: str = Field(..., description="Recognized character (A-Z, 0-9)")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    box: BoundingBox = Field(..., description="Character bounding box coordinates within plate")


class PlatePrediction(BaseModel):
    plate_number: str = Field(..., description="Full recognized license plate number string")
    plate_type: str = Field(..., description="Plate layout: 'MP' (Multi-row) or 'SP' (Single-row)")
    confidence: float = Field(..., description="Plate detector confidence score")
    plate_box: BoundingBox = Field(..., description="Coordinates of the plate in the original image")
    state_code: Optional[str] = Field(None, description="Extracted Indian state code (e.g., 'CG', 'AP', 'GJ', 'MH')")
    state_name: Optional[str] = Field(None, description="Full state name if resolved")
    is_valid_rto: bool = Field(False, description="True if plate code matches known RTO registry patterns")
    characters_count: int = Field(..., description="Number of recognized characters")
    characters: List[CharacterPrediction] = Field(default_factory=list, description="List of recognized characters and locations")
    cropped_image_base64: Optional[str] = Field(None, description="Base64 JPEG of the cropped plate image (if requested)")


class ANPRResponse(BaseModel):
    success: bool = Field(..., description="True if inference executed without error")
    message: str = Field(..., description="Human-readable result summary")
    total_plates: int = Field(..., description="Number of license plates detected in the image")
    primary_plate_number: Optional[str] = Field(None, description="Highest-confidence recognized plate number")
    plates: List[PlatePrediction] = Field(default_factory=list, description="Details for each plate found")
    annotated_image_base64: Optional[str] = Field(None, description="Base64 JPEG with bounding boxes and labels drawn (if requested)")
    processing_time_ms: float = Field(..., description="Inference latency in milliseconds")


class DirectOCRResponse(BaseModel):
    success: bool
    plate_number: str
    confidence: float
    characters_count: int
    characters: List[CharacterPrediction] = []
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str
    plate_model_loaded: bool
    ocr_model_loaded: bool
    models_path: Dict[str, str]


class JSONImageRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image string (with or without data:image/jpeg;base64, prefix)")
    image_url: Optional[str] = Field(None, description="Public URL to an image to fetch and process")
    conf_threshold: Optional[float] = Field(0.15, description="Plate detection confidence threshold (0.01 - 0.99)")
    ocr_conf_threshold: Optional[float] = Field(0.15, description="OCR character detection confidence threshold (0.01 - 0.99)")
    return_annotated_image: Optional[bool] = Field(False, description="Include base64 annotated image in response")
    return_crop: Optional[bool] = Field(False, description="Include base64 cropped plate image in response")


class KeyGenerateRequest(BaseModel):
    client_name: str = Field(..., description="Name of the client/user")
    description: Optional[str] = Field("", description="Optional purpose or description for the key")


class KeyInfo(BaseModel):
    key: str
    client: str
    created_at: str
    is_active: bool
    description: str
