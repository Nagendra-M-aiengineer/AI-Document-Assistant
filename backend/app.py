import os
import shutil
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from .rag import process_document, ask_question
    from .vector_store import (
        get_total_chunks,
        get_indexed_documents,
        delete_document,
        clear_all_documents,
    )
    from .config import (
        UPLOAD_FOLDER,
        get_runtime_settings,
        save_runtime_settings,
    )
except ImportError:
    from rag import process_document, ask_question
    from vector_store import (
        get_total_chunks,
        get_indexed_documents,
        delete_document,
        clear_all_documents,
    )
    from config import (
        UPLOAD_FOLDER,
        get_runtime_settings,
        save_runtime_settings,
    )

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
if not (FRONTEND_DIR / "index.html").exists():
    FRONTEND_DIR = BASE_DIR.parent
UPLOAD_DIR = Path(UPLOAD_FOLDER).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Document Assistant API",
    description="Full-stack Enterprise RAG with Vector Search, Citations, and Multi-LLM Support",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory for static assets
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ChatRequest(BaseModel):
    question: str
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None


class SettingsRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    groq_model: Optional[str] = None
    gemini_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    top_k: Optional[int] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def read_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_file)


@app.get("/style.css")
def get_style():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def get_app_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
def health():
    settings = get_runtime_settings()
    has_openai = bool(settings.get("openai_api_key"))
    has_groq = bool(settings.get("groq_api_key"))
    return {
        "status": "healthy",
        "total_chunks": get_total_chunks(),
        "total_documents": len(get_indexed_documents()),
        "active_provider": settings.get("provider", "auto"),
        "has_openai_key": has_openai,
        "has_groq_key": has_groq,
    }


@app.get("/documents")
def list_documents():
    docs = get_indexed_documents()
    # Augment with disk file sizes
    augmented = []
    for d in docs:
        fname = d["filename"]
        disk_file = UPLOAD_DIR / fname
        file_size = disk_file.stat().st_size if disk_file.exists() else 0
        augmented.append({
            "filename": fname,
            "chunks": d["chunks"],
            "total_pages": d["total_pages"],
            "size_bytes": file_size,
            "size_formatted": _format_size(file_size)
        })
    return {"documents": augmented, "total_chunks": get_total_chunks()}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files were uploaded")

    allowed_extensions = {".pdf", ".docx", ".txt", ".pptx", ".md", ".csv"}
    results = []

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}' in '{file.filename}'. Allowed: PDF, DOCX, TXT, PPTX, MD, CSV."
            )

        destination_path = UPLOAD_DIR / file.filename
        with destination_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            res = process_document(str(destination_path))
            results.append(res)
        except Exception as exc:
            # Clean up corrupted upload if failed
            if destination_path.exists():
                try:
                    destination_path.unlink()
                except Exception:
                    pass
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Processing failed for '{file.filename}': {str(exc)}") from exc

    return {
        "uploaded": results,
        "total_documents": len(get_indexed_documents()),
        "total_chunks": get_total_chunks(),
        "message": f"Successfully indexed {len(results)} file(s)."
    }


@app.post("/chat")
async def chat(payload: ChatRequest):
    q = payload.question.strip() if payload.question else ""
    if not q:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please enter a question.")

    result = ask_question(
        question=q,
        override_provider=payload.provider,
        override_key=payload.api_key,
        override_model=payload.model
    )
    return result


@app.delete("/documents/{filename}")
def remove_document(filename: str):
    clean_name = os.path.basename(filename)
    deleted_vectors = delete_document(clean_name)

    # Remove from uploads directory
    disk_file = UPLOAD_DIR / clean_name
    if disk_file.exists():
        try:
            disk_file.unlink()
        except Exception:
            pass

    return {
        "status": "success",
        "deleted": clean_name,
        "remaining_documents": len(get_indexed_documents()),
        "remaining_chunks": get_total_chunks()
    }


@app.post("/clear")
def clear_knowledge_base():
    clear_all_documents()
    # Clean uploaded files
    for item in UPLOAD_DIR.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except Exception:
                pass
    return {"status": "cleared", "total_chunks": 0, "total_documents": 0}


@app.get("/settings")
def get_settings():
    s = get_runtime_settings()
    # Mask API keys for security
    masked_openai = None
    if s.get("openai_api_key"):
        k = s["openai_api_key"]
        masked_openai = f"{k[:7]}...{k[-4:]}" if len(k) > 12 else "****"

    masked_groq = None
    if s.get("groq_api_key"):
        g = s["groq_api_key"]
        masked_groq = f"{g[:6]}...{g[-4:]}" if len(g) > 10 else "****"

    masked_gemini = None
    if s.get("gemini_api_key"):
        gm = s["gemini_api_key"]
        masked_gemini = f"{gm[:6]}...{gm[-4:]}" if len(gm) > 10 else "****"

    return {
        "provider": s.get("provider", "auto"),
        "model": s.get("model", "gpt-4o-mini"),
        "groq_model": s.get("groq_model", "llama-3.3-70b-versatile"),
        "gemini_model": s.get("gemini_model", "gemini-1.5-flash"),
        "chunk_size": s.get("chunk_size", 500),
        "chunk_overlap": s.get("chunk_overlap", 100),
        "top_k": s.get("top_k", 4),
        "has_openai_key": bool(s.get("openai_api_key")),
        "has_groq_key": bool(s.get("groq_api_key")),
        "has_gemini_key": bool(s.get("gemini_api_key")),
        "masked_openai_key": masked_openai,
        "masked_groq_key": masked_groq,
        "masked_gemini_key": masked_gemini,
    }


@app.post("/settings")
def update_settings(payload: SettingsRequest):
    updates = {}
    if payload.provider is not None:
        updates["provider"] = payload.provider
    if payload.model is not None:
        updates["model"] = payload.model
    if payload.groq_model is not None:
        updates["groq_model"] = payload.groq_model
    if payload.gemini_model is not None:
        updates["gemini_model"] = payload.gemini_model
    if payload.chunk_size is not None and payload.chunk_size > 50:
        updates["chunk_size"] = payload.chunk_size
    if payload.chunk_overlap is not None and payload.chunk_overlap >= 0:
        updates["chunk_overlap"] = payload.chunk_overlap
    if payload.top_k is not None and payload.top_k > 0:
        updates["top_k"] = payload.top_k
    if payload.openai_api_key is not None and payload.openai_api_key.strip():
        updates["openai_api_key"] = payload.openai_api_key.strip()
    if payload.groq_api_key is not None and payload.groq_api_key.strip():
        updates["groq_api_key"] = payload.groq_api_key.strip()
    if payload.gemini_api_key is not None and payload.gemini_api_key.strip():
        updates["gemini_api_key"] = payload.gemini_api_key.strip()

    saved = save_runtime_settings(updates)
    return {"status": "saved", "settings": saved}


def _format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} GB"
