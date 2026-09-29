# ANPR Model REST API Documentation

This REST API allows any external client, application, mobile app, or developer to use your **Automatic Number Plate Recognition (ANPR)** deep learning model by sending an HTTP request with your **API URL** and **API Key**.

---

## 1. Quick Overview

- **Local API URL:** `http://localhost:8000`
- **Network API URL:** `http://<SERVER_IP>:8000` (e.g. `http://192.168.1.9:8000`)
- **Interactive Swagger Docs:** `http://localhost:8000/docs`
- **Interactive ReDoc:** `http://localhost:8000/redoc`
- **Primary API Key:** `anpr_live_secret_key_2026`

---

## 2. Authentication (API Key)

Every request to the inference endpoints must include a valid **API Key**. You can pass the API key in any of the following 3 ways:

### Option A: Header `X-API-Key` (Recommended)
```http
X-API-Key: anpr_live_secret_key_2026
```

### Option B: Bearer Token Header
```http
Authorization: Bearer anpr_live_secret_key_2026
```

### Option C: Query Parameter
```http
http://localhost:8000/api/v1/anpr/predict?api_key=anpr_live_secret_key_2026
```

---

## 3. Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service information & documentation links |
| `GET` | `/api/v1/health` | Model status, GPU/CPU info, and health check |
| `POST` | `/api/v1/anpr/predict` | **Main Endpoint**: Upload an image file (Multipart form) |
| `POST` | `/api/v1/anpr/predict-json` | Send image via Base64 or remote Image URL |
| `POST` | `/api/v1/anpr/ocr-cropped` | Run OCR directly on an already-cropped plate |
| `GET` | `/api/v1/keys` | List all active API keys |
| `POST` | `/api/v1/keys/generate` | Generate a new API key for a client/user |

---

## 4. Usage Examples (How Anyone Can Use Your API)

### A. Python (`requests`)

```python
import requests

API_URL = "http://<SERVER_IP>:8000/api/v1/anpr/predict"
API_KEY = "anpr_live_secret_key_2026"

headers = {
    "X-API-Key": API_KEY
}

# 1. Send an image file
with open("car_image.jpg", "rb") as image_file:
    files = {"file": ("car_image.jpg", image_file, "image/jpeg")}
    params = {
        "conf_threshold": 0.15,
        "return_annotated_image": False,
        "return_crop": False
    }
    
    response = requests.post(API_URL, headers=headers, params=params, files=files)

data = response.json()
print("Plate Number:", data["primary_plate_number"])
print("Total Plates:", data["total_plates"])
print("Latency (ms):", data["processing_time_ms"])

for plate in data["plates"]:
    print(f"- Plate: {plate['plate_number']} ({plate['plate_type']}) | State: {plate['state_name']} | Conf: {plate['confidence']}")
```

---

### B. cURL (Command Line / Terminal)

```bash
# Using Header
curl -X POST "http://localhost:8000/api/v1/anpr/predict" \
     -H "X-API-Key: anpr_live_secret_key_2026" \
     -F "file=@path/to/vehicle.jpg"

# Or using Query Parameter
curl -X POST "http://localhost:8000/api/v1/anpr/predict?api_key=anpr_live_secret_key_2026" \
     -F "file=@path/to/vehicle.jpg"
```

---

### C. JavaScript / Node.js (`fetch`)

```javascript
const fs = require('fs');
const FormData = require('form-data');
const fetch = require('node-fetch');

async function detectPlate() {
  const form = new FormData();
  form.append('file', fs.createReadStream('car_image.jpg'));

  const response = await fetch('http://localhost:8000/api/v1/anpr/predict', {
    method: 'POST',
    headers: {
      'X-API-Key': 'anpr_live_secret_key_2026',
      ...form.getHeaders()
    },
    body: form
  });

  const result = await response.json();
  console.log('Recognized Plate:', result.primary_plate_number);
  console.log('Details:', result.plates);
}

detectPlate();
```

---

### D. JSON Base64 Endpoint (`/api/v1/anpr/predict-json`)

```bash
curl -X POST "http://localhost:8000/api/v1/anpr/predict-json" \
     -H "X-API-Key: anpr_live_secret_key_2026" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "/9j/4AAQSkZJRgABAQ...",
       "conf_threshold": 0.15,
       "return_annotated_image": false
     }'
```

---

## 5. Sample API Response

```json
{
  "success": true,
  "message": "Detected 1 plate(s)",
  "total_plates": 1,
  "primary_plate_number": "CG04MB1234",
  "plates": [
    {
      "plate_number": "CG04MB1234",
      "plate_type": "SP",
      "confidence": 0.885,
      "plate_box": {
        "x1": 312,
        "y1": 420,
        "x2": 520,
        "y2": 490,
        "width": 208,
        "height": 70
      },
      "state_code": "CG",
      "state_name": "Chhattisgarh",
      "is_valid_rto": true,
      "characters_count": 10,
      "characters": [
        {
          "char": "C",
          "confidence": 0.92,
          "box": { "x1": 10, "y1": 15, "x2": 28, "y2": 55, "width": 18, "height": 40 }
        },
        {
          "char": "G",
          "confidence": 0.94,
          "box": { "x1": 30, "y1": 15, "x2": 49, "y2": 55, "width": 19, "height": 40 }
        }
      ],
      "cropped_image_base64": null
    }
  ],
  "annotated_image_base64": null,
  "processing_time_ms": 142.5
}
```

---

## 6. How to Start the API Server

### Method 1: Double Click (Windows)
Double click `start_api.bat` in the project directory.

### Method 2: Command Line
```powershell
python run_api.py --port 8000
```

### Custom Port or Host:
```powershell
python run_api.py --host 0.0.0.0 --port 5000
```

---

## 7. How to Create New API Keys for Clients

You can create new API keys anytime using the API endpoint or Python script:

### Using Python:
```python
import requests

resp = requests.post(
    "http://localhost:8000/api/v1/keys/generate",
    headers={"X-API-Key": "anpr_live_secret_key_2026"},
    json={
        "client_name": "New Client Name",
        "description": "API Key for External App"
    }
)
print("New Key Generated:", resp.json())
```

All keys are stored in `config/api_keys.json` and persist across server reboots.
