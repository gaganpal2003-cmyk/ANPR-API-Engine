import os
import sys
import argparse
import socket
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure YOLO internal imports resolve
YOLO_DIR = ROOT_DIR / "Modules" / "ObjectDetectionModel" / "objectdetection"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

# Load environment variables from .env
ENV_FILE = ROOT_DIR / ".env"
try:
    from dotenv import load_dotenv
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)
except ImportError:
    pass

import configparser
import uvicorn
from api.security import get_all_active_keys


def get_local_ip() -> str:
    """Attempt to get local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner(host: str, port: int, api_key: str, local_ip: str):
    banner = f"""
================================================================================
               AUTOMATIC NUMBER PLATE RECOGNITION (ANPR) API
================================================================================
 [+] API Server Status:   RUNNING
 [+] Host:                {host}
 [+] Port:                {port}
 [+] Local URL:           http://localhost:{port}
 [+] Network URL:         http://{local_ip}:{port}
 [+] Interactive Docs:    http://localhost:{port}/docs
 [+] Health Check:        http://localhost:{port}/api/v1/health

 ----------------------------- API KEY INFORMATION -----------------------------
 [!] Primary API Key:     {api_key}
 [!] Header Name:         X-API-Key: {api_key}
 [!] Query Parameter:     ?api_key={api_key}

 --------------------------------- ENDPOINTS -----------------------------------
 1. POST  /api/v1/anpr/predict       -> File upload (Multipart/form-data)
 2. POST  /api/v1/anpr/predict-json  -> Base64 / URL JSON payload
 3. POST  /api/v1/anpr/ocr-cropped   -> Direct OCR on cropped plate
 4. GET   /api/v1/keys               -> List all active API keys
 5. POST  /api/v1/keys/generate      -> Create a new API key for a client

 [!] Share the Network URL and API Key with anyone who needs to use your model!
================================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="Run the ANPR REST API Server")
    parser.add_argument("--host", type=str, default=None, help="Host address (default from settings.ini or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default from settings.ini or 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    # Load configuration from .env or settings.ini
    default_host = os.getenv("ANPR_API_HOST", "0.0.0.0")
    default_port = int(os.getenv("ANPR_API_PORT", 8000))
    primary_key = os.getenv("ANPR_API_KEY", "anpr_live_secret_key_2026")

    settings_file = ROOT_DIR / "config" / "settings.ini"
    if settings_file.exists():
        try:
            config = configparser.ConfigParser()
            config.optionxform = str
            config.read(str(settings_file))
            if config.has_section("API"):
                default_host = os.getenv("ANPR_API_HOST") or config.get("API", "HOST", fallback=default_host)
                default_port = int(os.getenv("ANPR_API_PORT") or config.get("API", "PORT", fallback=str(default_port)))
                primary_key = os.getenv("ANPR_API_KEY") or config.get("API", "API_KEY", fallback=primary_key)
        except Exception as e:
            print(f"[Warning] Failed reading settings.ini: {e}")

    host = args.host or default_host
    port = args.port or default_port
    local_ip = get_local_ip()

    print_banner(host, port, primary_key, local_ip)

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=1
    )


if __name__ == "__main__":
    main()
