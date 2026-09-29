"""
FastAPI Main Application Entry Point
You can run this application using:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
or simply:
    python main.py
"""

import os
import sys
from pathlib import Path

# Add project root and YOLO internal imports to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

YOLO_DIR = ROOT_DIR / "Modules" / "ObjectDetectionModel" / "objectdetection"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

# Load .env
ENV_FILE = ROOT_DIR / ".env"
try:
    from dotenv import load_dotenv
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)
except ImportError:
    pass

import uvicorn
# Import the FastAPI application
from api.app import app

if __name__ == "__main__":
    host = os.getenv("ANPR_API_HOST", "0.0.0.0")
    # Render and cloud hosts inject the port in $PORT
    port = int(os.getenv("PORT") or os.getenv("ANPR_API_PORT") or 8000)
    is_dev = os.getenv("ENVIRONMENT", "development").lower() == "development"
    print(f"\n=======================================================")
    print(f"        Starting FastAPI ANPR Server on http://{host}:{port}")
    print(f"        Interactive Swagger Docs: http://localhost:{port}/docs")
    print(f"=======================================================\n")
    uvicorn.run("main:app", host=host, port=port, reload=is_dev)

