import json
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if os.getenv("VERCEL"):
    UPLOAD_FOLDER = Path("/tmp/uploads")
    CHROMA_DB_PATH = Path("/tmp/chroma_db")
    SETTINGS_PATH = Path("/tmp/runtime_settings.json")
else:
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    CHROMA_DB_PATH = BASE_DIR / "chroma_db"
    SETTINGS_PATH = BASE_DIR / "runtime_settings.json"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)


def _clean_key(key: str | None) -> str | None:
    if not key:
        return None
    key = str(key).strip().strip("'").strip('"')
    lower = key.lower()
    if lower.startswith("your_") or "example" in lower or "placeholder" in lower:
        return None
    return key


def get_runtime_settings() -> dict:
    defaults = {
        "provider": "auto",  # 'auto', 'openai', 'groq', 'gemini', 'grounded_synthesizer'
        "model": "gpt-4o-mini",
        "groq_model": "llama-3.3-70b-versatile",
        "gemini_model": "gemini-1.5-flash",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 4,
        "openai_api_key": None,
        "groq_api_key": None,
        "gemini_api_key": None,
    }

    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                saved = json.load(f)
                defaults.update({k: v for k, v in saved.items() if v is not None})
        except Exception:
            pass

    # Fallback to .env for keys if not in runtime_settings
    env_openai = os.getenv("OPENAI_API_KEY") or os.getenv("openai-api-key") or os.getenv("OPENAI_KEY")
    env_groq = os.getenv("GROQ_API_KEY") or os.getenv("groq-api-key")
    env_gemini = os.getenv("GEMINI_API_KEY") or os.getenv("gemini-api-key") or os.getenv("GOOGLE_API_KEY")

    if not defaults.get("openai_api_key") and env_openai:
        defaults["openai_api_key"] = _clean_key(env_openai)
    if not defaults.get("groq_api_key") and env_groq:
        defaults["groq_api_key"] = _clean_key(env_groq)
    if not defaults.get("gemini_api_key") and env_gemini:
        defaults["gemini_api_key"] = _clean_key(env_gemini)

    return defaults


def save_runtime_settings(updates: dict) -> dict:
    current = get_runtime_settings()
    for key, value in updates.items():
        if key in current:
            current[key] = value

    try:
        with SETTINGS_PATH.open("w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception as exc:
        print(f"Error saving runtime settings: {exc}")

    # Also update in-memory environment if API keys passed
    if updates.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = updates["openai_api_key"]
    if updates.get("groq_api_key"):
        os.environ["GROQ_API_KEY"] = updates["groq_api_key"]
    if updates.get("gemini_api_key"):
        os.environ["GEMINI_API_KEY"] = updates["gemini_api_key"]

    return current


# Default parameters
DEFAULT_SETTINGS = get_runtime_settings()
OPENAI_API_KEY = DEFAULT_SETTINGS.get("openai_api_key")
GROQ_API_KEY = DEFAULT_SETTINGS.get("groq_api_key")
GEMINI_API_KEY = DEFAULT_SETTINGS.get("gemini_api_key")
CHUNK_SIZE = DEFAULT_SETTINGS.get("chunk_size", 500)
CHUNK_OVERLAP = DEFAULT_SETTINGS.get("chunk_overlap", 100)
TOP_K = DEFAULT_SETTINGS.get("top_k", 4)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
