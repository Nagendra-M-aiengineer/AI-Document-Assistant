# 🧠 DocuSense AI Document Assistant (RAG)

Welcome to your first **Retrieval-Augmented Generation (RAG)** application! **DocuSense** allows you to upload any document (PDF, Word, PowerPoint, Markdown, Text, CSV) and ask questions in natural language. The AI answers grounded strictly in your documents with page citations and exact text evidence.

---

## 🌟 What is RAG? (How This App Works)

When working with Large Language Models (LLMs):
1. **The Problem**: Standard LLMs don't know your private documents, and their context window is limited.
2. **The RAG Solution**:
   - **Step 1 (Ingestion)**: You upload documents (`.pdf`, `.docx`, `.txt`, `.pptx`, `.md`, `.csv`).
   - **Step 2 (Chunking)**: `RecursiveCharacterTextSplitter` breaks long documents into coherent semantic passages with overlap.
   - **Step 3 (Embeddings)**: `sentence-transformers/all-MiniLM-L6-v2` converts every chunk into high-dimensional vector representations.
   - **Step 4 (Vector Database)**: Vector embeddings and metadata (page numbers, filenames) are stored in **ChromaDB**.
   - **Step 5 (Semantic Retrieval)**: When you ask a question, the question is embedded and Chroma retrieves the top-$k$ most relevant context excerpts.
   - **Step 6 (Augmented Generation)**: The retrieved excerpts and your question are passed to the LLM (OpenAI GPT-4o / Groq Llama 3 / Local Grounded Synthesizer) with strict anti-hallucination instructions.
   - **Step 7 (Grounded Output & Citations)**: The answer is rendered in Markdown with expandable source cards citing the exact source file, page number, and confidence score.

---

## 🚀 Quick Start

### 1. Launch the Server

Double-click `run.bat` (on Windows) or run in your terminal:

```bash
python run.py
```

### 2. Open the Web Application

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

API interactive documentation is available at:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## 🎯 Key Features

- **Multi-Format Document Parsing**:
  - PDFs (with PyPDF & PyMuPDF fallbacks)
  - Microsoft Word (`.docx`)
  - PowerPoint presentations (`.pptx`)
  - Text & Markdown (`.txt`, `.md`)
  - Spreadsheets (`.csv`)
- **Modern Enterprise Themed UI**:
  - Dark mode and Light mode with smooth toggle
  - Sleek glassmorphism topbar, floating composer, and responsive sidebar
  - Document manager showing indexed files, chunk counts, and file sizes
  - Real-time drag-and-drop file upload with animated progress
  - One-click quick starter prompt chips
- **Accurate Citations & Source Evidence**:
  - Every answer includes an expandable **"Referenced Sources"** drawer
  - Shows file name, page/slide number, match confidence percentage, and exact text excerpts
- **Resilient Multi-LLM Architecture**:
  - **OpenAI**: Supports `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`
  - **Groq**: Supports `llama-3.3-70b-versatile` (ultra-fast free tier from Groq)
  - **Built-in Grounded Semantic Extractor**: If your OpenAI key has run out of credits (Error 429) or you are offline, DocuSense **never crashes** — it intelligently synthesizes an answer directly from the retrieved document excerpts!
- **Dynamic In-App Settings**:
  - Configure or change API Keys directly in the UI settings dialog without editing code or restarting the server.
  - Adjust chunk size, overlap, and number of retrieved passages (Top-K).

---

## 📁 Project Structure

```
AI Document Assistant/
│
├── backend/
│   ├── app.py               # FastAPI backend with REST endpoints
│   ├── config.py            # Environment, path, and runtime settings
│   ├── document_loader.py   # Multi-format document loaders
│   ├── embeddings.py        # HuggingFace & fallback embeddings
│   ├── rag.py               # Retrieval & Generation pipeline
│   ├── vector_store.py      # ChromaDB storage & similarity search
│   ├── uploads/             # Storage for uploaded files
│   ├── chroma_db/           # Persistent vector database
│   └── .env                 # Environment variables (API keys)
│
├── frontend/
│   ├── index.html           # Modern responsive layout
│   ├── style.css            # Dark/light theme styles & glassmorphism
│   └── app.js               # Reactive client application logic
│
├── run.py                   # One-command startup script
├── run.bat                  # Windows 1-click launcher
└── README.md                # Documentation & Architecture guide
```

---

## 🔑 Configuring Your API Keys

You can configure your keys in two ways:

1. **Through the UI (Recommended)**:
   - Click the **Gear icon (⚙️)** in the top right of the application.
   - Enter your OpenAI or Groq API key and click **Save Settings**.
2. **Via `.env` file**:
   - Open `backend/.env` and update:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     GROQ_API_KEY=your_groq_api_key_here
     ```
