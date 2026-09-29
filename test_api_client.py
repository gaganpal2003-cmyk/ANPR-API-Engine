"""
ANPR API Client Demo Script
Demonstrates how any client or developer can use the ANPR API URL and API Key
to recognize number plates from images.
"""

import os
import sys
import json
import base64
from pathlib import Path
import requests

# Load environment variables from .env file
ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
try:
    from dotenv import load_dotenv
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)
except ImportError:
    pass

# Configuration loaded from .env (with safe fallbacks)
API_URL = os.getenv("ANPR_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("ANPR_API_KEY", "anpr_live_secret_key_2026")

# Test sample image
SAMPLE_IMAGE = ROOT_DIR / "anprPicture" / "Camera_1" / "temp_cropped.jpg"


def test_health():
    print("\n--- 1. Testing Health Check ---")
    url = f"{API_URL}/api/v1/health"
    try:
        resp = requests.get(url, timeout=5)
        print("Status Code:", resp.status_code)
        print("Response:", json.dumps(resp.json(), indent=2))
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API server at {API_URL}. Is run_api.py running?")
        return False


def test_auth_failure():
    print("\n--- 2. Testing Authentication Failure (Invalid Key) ---")
    url = f"{API_URL}/api/v1/anpr/predict"
    headers = {"X-API-Key": "invalid_wrong_key"}
    resp = requests.post(url, headers=headers)
    print("Status Code:", resp.status_code, "(Expected: 403 or 401)")
    print("Response:", resp.json())


def test_predict_file_upload():
    print("\n--- 3. Testing Predict via File Upload (Multipart Form) ---")
    if not SAMPLE_IMAGE.exists():
        print(f"Sample image not found at: {SAMPLE_IMAGE}")
        return

    url = f"{API_URL}/api/v1/anpr/predict"
    headers = {
        "X-API-Key": API_KEY
    }
    params = {
        "conf_threshold": 0.15,
        "return_annotated_image": False,
        "return_crop": False
    }

    with open(SAMPLE_IMAGE, "rb") as f:
        files = {"file": ("plate.jpg", f, "image/jpeg")}
        resp = requests.post(url, headers=headers, params=params, files=files)

    print("Status Code:", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        print("\nRecognition Result:")
        print(f" - Primary Plate Number: {data.get('primary_plate_number')}")
        print(f" - Total Plates Detected: {data.get('total_plates')}")
        print(f" - Latency: {data.get('processing_time_ms')} ms")
        for i, plate in enumerate(data.get("plates", []), 1):
            print(f"\n   Plate #{i}:")
            print(f"   • Number:     {plate['plate_number']}")
            print(f"   • Type:       {plate['plate_type']}")
            print(f"   • State:      {plate['state_code']} ({plate['state_name']})")
            print(f"   • RTO Valid:  {plate['is_valid_rto']}")
            print(f"   • Confidence: {plate['confidence']}")
            print(f"   • Characters: {''.join(c['char'] for c in plate['characters'])}")
    else:
        print("Error Response:", resp.text)


def test_predict_base64():
    print("\n--- 4. Testing Predict via Base64 JSON Payload ---")
    if not SAMPLE_IMAGE.exists():
        return

    with open(SAMPLE_IMAGE, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode("utf-8")

    url = f"{API_URL}/api/v1/anpr/predict-json"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "image_base64": b64_str,
        "conf_threshold": 0.15
    }

    resp = requests.post(url, headers=headers, json=payload)
    print("Status Code:", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Recognized Plate: {data.get('primary_plate_number')} in {data.get('processing_time_ms')} ms")
    else:
        print("Error:", resp.text)


def test_direct_ocr():
    print("\n--- 5. Testing Direct OCR on Cropped Plate ---")
    if not SAMPLE_IMAGE.exists():
        return

    url = f"{API_URL}/api/v1/anpr/ocr-cropped"
    headers = {"X-API-Key": API_KEY}
    params = {"plate_type": "MP"}

    with open(SAMPLE_IMAGE, "rb") as f:
        files = {"file": ("crop.jpg", f, "image/jpeg")}
        resp = requests.post(url, headers=headers, params=params, files=files)

    print("Status Code:", resp.status_code)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Direct OCR Plate: {data.get('plate_number')} (conf: {data.get('confidence')})")
    else:
        print("Error:", resp.text)


if __name__ == "__main__":
    print(f"Connecting to ANPR API at: {API_URL}")
    print(f"Using API Key: {API_KEY}")

    is_alive = test_health()
    if is_alive:
        test_auth_failure()
        test_predict_file_upload()
        test_predict_base64()
        test_direct_ocr()
    else:
        print("\nPlease start the server first in another terminal with: python run_api.py")
