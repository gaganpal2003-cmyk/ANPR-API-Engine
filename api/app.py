import io
import time
import base64
import urllib.request
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Depends, File, UploadFile, Query, Form, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api.security import get_current_api_key, generate_new_api_key, load_keys_from_registry, get_all_active_keys
from api.schemas import (
    ANPRResponse,
    PlatePrediction,
    BoundingBox,
    CharacterPrediction,
    DirectOCRResponse,
    HealthResponse,
    JSONImageRequest,
    KeyGenerateRequest,
    KeyInfo
)
from api.engine import ANPRInferenceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ANPR deep learning models into memory/GPU
    print("[API Server] Starting up... Initializing ANPR Deep Learning Models...")
    engine = ANPRInferenceEngine.get_instance()
    app.state.engine = engine
    print(f"[API Server] Ready! Serving on device: {engine.device}")
    yield
    # Shutdown
    print("[API Server] Shutting down...")


app = FastAPI(
    title="ANPR Model REST API",
    description="""
## Automatic Number Plate Recognition (ANPR) API

This API enables you to perform state-of-the-art license plate detection and optical character recognition (OCR) on vehicle images.

### Authentication
You must provide your **API Key** in one of the following ways:
1. **Header**: `X-API-Key: <your_api_key>`
2. **Bearer Token**: `Authorization: Bearer <your_api_key>`
3. **Query Parameter**: `?api_key=<your_api_key>`

Click the **Authorize** button in the top-right corner to test the API directly from this documentation page!
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes to OpenCV BGR numpy array."""
    if not data:
        raise HTTPException(status_code=400, detail="Empty image data provided.")
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please ensure valid JPG, PNG, or WEBP format.")
    return img


def _read_image_from_base64(b64_str: str) -> np.ndarray:
    """Decode base64 string (handles data URL prefixes)."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        data = base64.b64decode(b64_str)
        return _read_image_from_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")


def _read_image_from_url(url: str) -> np.ndarray:
    """Fetch remote image from URL and decode."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ANPR-Client/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return _read_image_from_bytes(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL '{url}': {str(e)}")


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/", tags=["General"])
async def root():
    """Welcome page with documentation links."""
    return {
        "service": "ANPR Model REST API",
        "version": "1.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health_check": "/api/v1/health",
        "predict_endpoint": "/api/v1/anpr/predict"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Check service health and model loading status."""
    engine = ANPRInferenceEngine.get_instance()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=str(engine.device),
        plate_model_loaded=engine.plate_model is not None,
        ocr_model_loaded=engine.ocr_model is not None,
        models_path={
            "plate_model": engine.plate_model_path,
            "ocr_model": engine.ocr_model_path
        }
    )


@app.post(
    "/api/v1/anpr/predict",
    response_model=ANPRResponse,
    tags=["Inference"],
    summary="Recognize Number Plate from File Upload"
)
async def predict_plate_from_file(
    file: UploadFile = File(..., description="Vehicle or license plate image file (JPG, PNG, WEBP)"),
    conf_threshold: Optional[float] = Query(0.15, ge=0.01, le=0.99, description="Plate detection confidence threshold"),
    ocr_conf_threshold: Optional[float] = Query(0.15, ge=0.01, le=0.99, description="OCR character confidence threshold"),
    return_annotated_image: bool = Query(False, description="Whether to include annotated base64 image"),
    return_crop: bool = Query(False, description="Whether to include cropped plate base64 image"),
    api_key: str = Depends(get_current_api_key)
):
    """
    **Primary ANPR Endpoint (File Upload)**
    - Upload an image file of a vehicle or license plate.
    - Performs plate detection, character recognition (OCR), row ordering, and Indian RTO state validation.
    - Requires valid API Key.
    """
    contents = await file.read()
    image_bgr = _read_image_from_bytes(contents)

    engine = ANPRInferenceEngine.get_instance()
    result = engine.process_image(
        image_bgr=image_bgr,
        conf_thresh=conf_threshold,
        ocr_conf_thresh=ocr_conf_threshold,
        return_annotated=return_annotated_image,
        return_crop=return_crop
    )

    # Format into Pydantic schema
    formatted_plates = []
    for p in result["plates"]:
        chars = [
            CharacterPrediction(
                char=c["char"],
                confidence=round(c["conf"], 4),
                box=BoundingBox(
                    x1=c["box"][0],
                    y1=c["box"][1],
                    x2=c["box"][2],
                    y2=c["box"][3],
                    width=c["box"][2] - c["box"][0],
                    height=c["box"][3] - c["box"][1]
                )
            ) for c in p["characters"]
        ]
        box = p["plate_box"]
        formatted_plates.append(
            PlatePrediction(
                plate_number=p["plate_number"],
                plate_type=p["plate_type"],
                confidence=round(p["confidence"], 4),
                plate_box=BoundingBox(
                    x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                    width=box[2] - box[0], height=box[3] - box[1]
                ),
                state_code=p["state_code"],
                state_name=p["state_name"],
                is_valid_rto=p["is_valid_rto"],
                characters_count=len(chars),
                characters=chars,
                cropped_image_base64=p.get("cropped_image_base64")
            )
        )

    return ANPRResponse(
        success=True,
        message=result["message"],
        total_plates=result["total_plates"],
        primary_plate_number=result["primary_plate_number"],
        plates=formatted_plates,
        annotated_image_base64=result.get("annotated_image_base64"),
        processing_time_ms=result["processing_time_ms"]
    )


@app.post(
    "/api/v1/anpr/predict-json",
    response_model=ANPRResponse,
    tags=["Inference"],
    summary="Recognize Number Plate from Base64 or Image URL"
)
async def predict_plate_from_json(
    request: JSONImageRequest,
    api_key: str = Depends(get_current_api_key)
):
    """
    **ANPR Endpoint (JSON Payload)**
    - Send an image as a **base64 string** or provide a public **image URL**.
    - Ideal for microservices, mobile apps, and programmatic integration.
    """
    if request.image_base64:
        image_bgr = _read_image_from_base64(request.image_base64)
    elif request.image_url:
        image_bgr = _read_image_from_url(request.image_url)
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'image_base64' or 'image_url' must be provided in request body."
        )

    engine = ANPRInferenceEngine.get_instance()
    result = engine.process_image(
        image_bgr=image_bgr,
        conf_thresh=request.conf_threshold,
        ocr_conf_thresh=request.ocr_conf_threshold,
        return_annotated=request.return_annotated_image or False,
        return_crop=request.return_crop or False
    )

    formatted_plates = []
    for p in result["plates"]:
        chars = [
            CharacterPrediction(
                char=c["char"],
                confidence=round(c["conf"], 4),
                box=BoundingBox(
                    x1=c["box"][0],
                    y1=c["box"][1],
                    x2=c["box"][2],
                    y2=c["box"][3],
                    width=c["box"][2] - c["box"][0],
                    height=c["box"][3] - c["box"][1]
                )
            ) for c in p["characters"]
        ]
        box = p["plate_box"]
        formatted_plates.append(
            PlatePrediction(
                plate_number=p["plate_number"],
                plate_type=p["plate_type"],
                confidence=round(p["confidence"], 4),
                plate_box=BoundingBox(
                    x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                    width=box[2] - box[0], height=box[3] - box[1]
                ),
                state_code=p["state_code"],
                state_name=p["state_name"],
                is_valid_rto=p["is_valid_rto"],
                characters_count=len(chars),
                characters=chars,
                cropped_image_base64=p.get("cropped_image_base64")
            )
        )

    return ANPRResponse(
        success=True,
        message=result["message"],
        total_plates=result["total_plates"],
        primary_plate_number=result["primary_plate_number"],
        plates=formatted_plates,
        annotated_image_base64=result.get("annotated_image_base64"),
        processing_time_ms=result["processing_time_ms"]
    )


@app.post(
    "/api/v1/anpr/ocr-cropped",
    response_model=DirectOCRResponse,
    tags=["Inference"],
    summary="Direct OCR on Cropped License Plate"
)
async def ocr_cropped_plate(
    file: UploadFile = File(..., description="Cropped license plate image"),
    ocr_conf_threshold: Optional[float] = Query(0.15, ge=0.01, le=0.99),
    plate_type: str = Query("SP", regex="^(SP|MP)$", description="'SP' (single-row) or 'MP' (multi-row)"),
    api_key: str = Depends(get_current_api_key)
):
    """
    Direct OCR endpoint for images that are already cropped plates.
    """
    t0 = time.time()
    contents = await file.read()
    image_bgr = _read_image_from_bytes(contents)

    engine = ANPRInferenceEngine.get_instance()
    ocr_res = engine.run_ocr_on_plate(image_bgr, plate_type=plate_type, ocr_conf_thresh=ocr_conf_threshold)

    chars = [
        CharacterPrediction(
            char=c["char"],
            confidence=round(c["conf"], 4),
            box=BoundingBox(
                x1=c["box"][0],
                y1=c["box"][1],
                x2=c["box"][2],
                y2=c["box"][3],
                width=c["box"][2] - c["box"][0],
                height=c["box"][3] - c["box"][1]
            )
        ) for c in ocr_res["characters"]
    ]

    return DirectOCRResponse(
        success=True,
        plate_number=ocr_res["plate_number"],
        confidence=ocr_res["confidence"],
        characters_count=len(chars),
        characters=chars,
        processing_time_ms=round((time.time() - t0) * 1000, 2)
    )


# =====================================================================
# API Key Management Endpoints
# =====================================================================

@app.get(
    "/api/v1/keys",
    response_model=List[KeyInfo],
    tags=["API Key Management"],
    summary="List All Active API Keys"
)
async def list_keys(api_key: str = Depends(get_current_api_key)):
    """List all registered API keys. Requires an authorized API key."""
    active_keys = get_all_active_keys()
    return [
        KeyInfo(
            key=info["key"],
            client=info.get("client", "Unknown"),
            created_at=str(info.get("created_at", "N/A")),
            is_active=info.get("is_active", True),
            description=info.get("description", "")
        )
        for info in active_keys.values()
    ]


@app.post(
    "/api/v1/keys/generate",
    response_model=KeyInfo,
    tags=["API Key Management"],
    summary="Generate a New API Key for a Client"
)
async def generate_key(
    req: KeyGenerateRequest,
    api_key: str = Depends(get_current_api_key)
):
    """
    Generate a new API key for any user/client so they can access your model.
    The new key is automatically saved to `config/api_keys.json` and activated immediately.
    """
    record = generate_new_api_key(client_name=req.client_name, description=req.description)
    return KeyInfo(
        key=record["key"],
        client=record["client"],
        created_at=record["created_at"],
        is_active=record["is_active"],
        description=record["description"]
    )
