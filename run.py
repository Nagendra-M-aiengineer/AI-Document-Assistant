import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
BACKEND_DIR = PROJECT_ROOT / "backend"

# Ensure python path in current process
paths = [str(VENV_SITE_PACKAGES), str(BACKEND_DIR), str(PROJECT_ROOT)]
for p in paths:
    if p not in sys.path:
        sys.path.insert(0, p)

# Set PYTHONPATH environment variable for any sub-processes
current_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([current_pp] if current_pp else []))

if __name__ == "__main__":
    print("\n" + "=" * 60, flush=True)
    print(" 🚀 Initializing DocuSense AI Document Assistant...", flush=True)
    print("=" * 60, flush=True)

    import uvicorn
    from backend.app import app

    print("\n" + "=" * 60, flush=True)
    print(" 🌟 Server is ready and running!", flush=True)
    print(" 🌐 Application URL: http://127.0.0.1:8000", flush=True)
    print(" 📄 API Docs URL:    http://127.0.0.1:8000/docs", flush=True)
    print("=" * 60 + "\n", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
