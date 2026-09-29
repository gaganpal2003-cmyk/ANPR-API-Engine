# ANPR API Engine

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7.svg)](https://render.com/)

A production-ready, high-performance **Automatic Number Plate Recognition (ANPR)** REST API microservice built with **FastAPI**, **PyTorch**, and **YOLO**. 

Enables developers, mobile apps, and camera systems to detect vehicle number plates and extract plate characters via a secure REST API with API Key authentication.

---

## 🚀 Features

- **End-to-End ANPR Pipeline**:
  - **Plate Detection**: Identifies license plate bounding boxes on vehicles using YOLO.
  - **Optical Character Recognition (OCR)**: Recognizes Indian vehicle registration numbers with state code mapping.
- **RESTful Endpoints**:
  - Multipart file upload (`POST /api/v1/anpr/predict`)
  - Base64 JSON and remote URL ingestion (`POST /api/v1/anpr/predict-json`)
  - Direct OCR for pre-cropped license plates (`POST /api/v1/anpr/ocr-cropped`)
  - Key management (`GET /api/v1/keys`, `POST /api/v1/keys/generate`)
  - System health check (`GET /api/v1/health`)
- **API Key Security**: Supports authentication via `X-API-Key` header, `Bearer` token, or query parameters.
- **Interactive Documentation**: Auto-generated Swagger UI (`/docs`) and ReDoc (`/redoc`).
- **Cloud Ready**: Pre-configured with `Dockerfile` and `render.yaml` for instant deployment on [Render](https://render.com).

---

## 📁 Repository Structure

```text
├── api/                        # FastAPI service application
│   ├── app.py                  # API endpoints and middleware
│   ├── engine.py               # Deep learning inference pipeline
│   ├── schemas.py              # Pydantic request/response schemas
│   └── security.py             # API Key authentication & verification
├── config/                     # Configuration files & registries
│   ├── settings.ini            # Server and inference threshold settings
│   ├── api_keys.json           # API key registry
│   └── RTO_CODES.txt           # Indian state & RTO code mappings
├── Detection/                  # Neural network models and labels
│   ├── model/
│   │   ├── best_plate.pt       # YOLO plate detector weights (~14 MB)
│   │   └── best_ocr.pt         # YOLO character OCR weights (~14 MB)
│   └── data/coco.yaml          # Class mapping
├── Modules/                    # Object detection model backend
│   └── ObjectDetectionModel/   # YOLO backend utilities
├── anprPicture/                # Sample test images
├── Dockerfile                  # Container definition for Linux/Render
├── render.yaml                 # Render Blueprint configuration
├── requirements.txt            # Python dependencies (CPU PyTorch + OpenCV)
├── main.py                     # Application entry point
├── run_api.py                  # CLI runner with formatted console banner
├── test_api_client.py          # Python client test script
├── API_DOCUMENTATION.md        # Comprehensive API documentation
└── RENDER_DEPLOYMENT_GUIDE.md  # Step-by-step Render deployment guide
```

---

## ⚡ Quick Start (Local Setup)

### 1. Install Dependencies

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the API Server

```bash
python main.py
```
*(Alternatively, on Windows double-click `start_api.bat` or run `python run_api.py`)*

The server will be running at:
- **API URL**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

---

## 🔑 Authentication

Include your API Key in requests via header:
```http
X-API-Key: anpr_live_secret_key_2026
```

---

## 🧪 Testing with Python

Run the built-in test client:
```bash
python test_api_client.py
```

Or perform inference using `requests`:
```python
import requests

url = "http://localhost:8000/api/v1/anpr/predict"
headers = {"X-API-Key": "anpr_live_secret_key_2026"}

with open("car_image.jpg", "rb") as f:
    response = requests.post(url, headers=headers, files={"file": f})

print(response.json())
```

---

## ☁️ Deployment on Render

This repository is optimized for [Render](https://render.com) Docker deployment.

1. Push this repository to GitHub.
2. Log in to [Render Dashboard](https://dashboard.render.com).
3. Create a **New Web Service** and select this repository.
4. Set **Runtime** to **Docker** and choose your plan.
5. Click **Create Web Service**.

For comprehensive deployment instructions, read [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md).

---

## 📄 Documentation

For full details on API endpoints, request schemas, and responses, refer to [API_DOCUMENTATION.md](API_DOCUMENTATION.md).
