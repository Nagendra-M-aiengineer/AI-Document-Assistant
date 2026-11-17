import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    filename = path.name

    if suffix == ".pdf":
        return _load_pdf(file_path, filename)
    elif suffix == ".txt" or suffix == ".md":
        return _load_text(file_path, filename)
    elif suffix == ".docx":
        return _load_docx(file_path, filename)
    elif suffix == ".pptx":
        return _load_pptx(file_path, filename)
    elif suffix == ".csv":
        return _load_csv(file_path, filename)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .pdf, .docx, .txt, .pptx, .md, .csv")


def _load_pdf(file_path: str, filename: str) -> List[Document]:
    # Try LangChain PyPDFLoader first
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if docs and any(d.page_content.strip() for d in docs):
            for doc in docs:
                doc.metadata["filename"] = filename
                doc.metadata["source"] = filename
            return docs
    except Exception:
        pass

    # Try PyMuPDF (fitz) as robust fallback
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        docs = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "filename": filename, "page": page_num + 1}
                ))
        doc.close()
        if docs:
            return docs
    except Exception:
        pass

    # Fallback to direct pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename, "filename": filename, "page": i + 1}
                ))
        if docs:
            return docs
    except Exception as exc:
        raise ValueError(f"Failed to read PDF file '{filename}': {exc}")

    return [Document(page_content="[Empty PDF file or unreadable scanned pages]", metadata={"source": filename, "filename": filename, "page": 1})]


def _load_text(file_path: str, filename: str) -> List[Document]:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": filename, "filename": filename, "page": 1})]
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise ValueError(f"Failed to read text file '{filename}': {exc}")

    raise ValueError(f"Could not decode text file '{filename}' with standard encodings.")


def _load_docx(file_path: str, filename: str) -> List[Document]:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["filename"] = filename
            doc.metadata["source"] = filename
        return docs
    except Exception:
        # Fallback to python-docx directly
        try:
            import docx
            doc = docx.Document(file_path)
            full_text = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        full_text.append(row_text)
            text = "\n\n".join(full_text)
            return [Document(page_content=text, metadata={"source": filename, "filename": filename, "page": 1})]
        except Exception as exc:
            raise ValueError(f"Failed to read DOCX file '{filename}': {exc}")


def _load_pptx(file_path: str, filename: str) -> List[Document]:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise ImportError("python-pptx is required to process PPTX files") from exc

    try:
        presentation = Presentation(file_path)
        docs = []
        for index, slide in enumerate(presentation.slides, start=1):
            text_parts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text_parts.append(shape.text.strip())
            if text_parts:
                docs.append(Document(
                    page_content="\n".join(text_parts),
                    metadata={"source": filename, "filename": filename, "page": index, "slide": index}
                ))

        if not docs:
            docs = [Document(page_content="[No text content found in presentation]", metadata={"source": filename, "filename": filename, "page": 1})]

        return docs
    except Exception as exc:
        raise ValueError(f"Failed to read PPTX file '{filename}': {exc}")


def _load_csv(file_path: str, filename: str) -> List[Document]:
    import csv
    encodings = ["utf-8", "latin-1"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                reader = csv.reader(f)
                rows = [", ".join(row) for row in reader if any(row)]
            text = "\n".join(rows)
            return [Document(page_content=text, metadata={"source": filename, "filename": filename, "page": 1})]
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file '{filename}': {exc}")
    raise ValueError(f"Could not decode CSV file '{filename}'.")
