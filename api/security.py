import os
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPBearer, HTTPAuthorizationCredentials

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.ini"
API_KEYS_FILE = CONFIG_DIR / "api_keys.json"
ENV_FILE = BASE_DIR / ".env"

try:
    from dotenv import load_dotenv
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE)
except ImportError:
    pass

# Header and query schemes for OpenAPI Swagger UI integration
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Provide your API Key in X-API-Key header")
api_key_query = APIKeyQuery(name="api_key", auto_error=False, description="Provide your API Key in query parameter (?api_key=...)")
bearer_scheme = HTTPBearer(auto_error=False, description="Provide your API Key as Bearer token")


def load_keys_from_registry() -> List[Dict]:
    """Load all keys and metadata from api_keys.json."""
    if not API_KEYS_FILE.exists():
        return []
    try:
        with open(API_KEYS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("keys", [])
    except Exception as e:
        print(f"[Security] Warning reading {API_KEYS_FILE}: {e}")
        return []


def save_keys_to_registry(keys_list: List[Dict]):
    """Save keys list to api_keys.json."""
    try:
        with open(API_KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump({"keys": keys_list}, f, indent=2)
    except Exception as e:
        print(f"[Security] Error saving {API_KEYS_FILE}: {e}")


def get_all_active_keys() -> Dict[str, Dict]:
    """
    Returns a dict mapping key_string -> key_info_dict
    Combines keys from api_keys.json and settings.ini.
    """
    active_keys = {}

    # 1. Load from .env if set
    env_key = os.getenv("ANPR_API_KEY", "").strip()
    if env_key:
        active_keys[env_key] = {
            "key": env_key,
            "client": ".env Configuration Key",
            "created_at": "environment",
            "is_active": True,
            "description": "Configured in .env file"
        }

    # 2. Load from api_keys.json
    for item in load_keys_from_registry():
        if item.get("is_active", True):
            active_keys[item["key"]] = item

    # 3. Load from settings.ini as well
    if SETTINGS_FILE.exists():
        try:
            import configparser
            config = configparser.ConfigParser()
            config.optionxform = str
            config.read(str(SETTINGS_FILE))
            if config.has_section("API"):
                ini_key = config.get("API", "API_KEY", fallback="").strip()
                if ini_key and ini_key not in active_keys:
                    active_keys[ini_key] = {
                        "key": ini_key,
                        "client": "Primary settings.ini Key",
                        "created_at": "system",
                        "is_active": True,
                        "description": "Configured in config/settings.ini"
                    }
        except Exception as e:
            print(f"[Security] Warning reading {SETTINGS_FILE}: {e}")

    # Fallback default key if none configured
    if not active_keys:
        default_key = "anpr_live_secret_key_2026"
        active_keys[default_key] = {
            "key": default_key,
            "client": "Default Admin Key",
            "created_at": str(datetime.now().date()),
            "is_active": True,
            "description": "Auto-generated default key"
        }

    return active_keys


def verify_api_key(provided_key: Optional[str]) -> bool:
    """Check if provided API key is valid and active."""
    if not provided_key:
        return False
    active = get_all_active_keys()
    return provided_key.strip() in active


def generate_new_api_key(client_name: str, description: str = "") -> Dict:
    """Generate a new secure API key, register it in api_keys.json, and return the record."""
    random_hex = secrets.token_hex(16)
    new_key = f"anpr_{random_hex}"
    record = {
        "key": new_key,
        "client": client_name.strip() or "Unnamed Client",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_active": True,
        "description": description.strip()
    }

    all_keys = load_keys_from_registry()
    all_keys.append(record)
    save_keys_to_registry(all_keys)
    return record


async def get_current_api_key(
    request: Request,
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query),
    bearer_auth: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> str:
    """
    FastAPI dependency to extract and validate the API Key.
    Accepts:
      1. 'X-API-Key' header
      2. 'Authorization: Bearer <key>' header
      3. '?api_key=<key>' query parameter
    """
    key_candidate = None

    if header_key:
        key_candidate = header_key.strip()
    elif bearer_auth and bearer_auth.credentials:
        key_candidate = bearer_auth.credentials.strip()
    elif query_key:
        key_candidate = query_key.strip()

    if not key_candidate:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing API Key. Please provide an API key using the 'X-API-Key' header, "
                "'Authorization: Bearer <your_key>' header, or query parameter '?api_key=<your_key>'."
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not verify_api_key(key_candidate):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or deactivated API Key. Access denied.",
        )

    return key_candidate
