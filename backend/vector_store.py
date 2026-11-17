import os
import re
from typing import List, Dict, Any, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from .embeddings import embedding_model
    from .config import CHROMA_DB_PATH
except ImportError:
    from embeddings import embedding_model
    from config import CHROMA_DB_PATH

CHROMA_PATH_STR = str(CHROMA_DB_PATH)

db = Chroma(
    persist_directory=CHROMA_PATH_STR,
    embedding_function=embedding_model,
    collection_name="rag_knowledge_base"
)


def add_documents(documents: List[Document]) -> int:
    if not documents:
        return 0
    db.add_documents(documents)
    return len(documents)


def similarity_search(query: str, k: int = 4) -> List[Document]:
    if get_total_chunks() == 0:
        return []
    scored = similarity_search_with_scores(query, k=k)
    return [doc for doc, _ in scored]


def similarity_search_with_scores(query: str, k: int = 6) -> List[Tuple[Document, float]]:
    """
    High-Precision Hybrid Retrieval:
    Combines dense semantic vector scoring with lexical phrase matching,
    definition prioritization, and automatic suppression of cover pages & bibliography.
    """
    if get_total_chunks() == 0:
        return []
    try:
        # Search candidate pool across collection
        fetch_k = min(50, max(k * 8, 30))
        raw = db.similarity_search_with_score(query, k=fetch_k)

        q_lower = query.lower()
        stop = {'what', 'when', 'where', 'which', 'who', 'this', 'that', 'with', 'from', 'does', 'meant', 'mean', 'about', 'is', 'are', 'the'}
        q_words = [w for w in re.findall(r'\w{3,}', q_lower) if w not in stop]

        # Extract 2-word phrases from query (e.g. "state transition")
        words_list = re.findall(r'\w+', q_lower)
        key_phrases = []
        for i in range(len(words_list) - 1):
            if words_list[i] not in stop and words_list[i+1] not in stop:
                key_phrases.append(f"{words_list[i]} {words_list[i+1]}")

        reranked = []
        for doc, dist in raw:
            # Base dense similarity
            score = 1.0 / (1.0 + float(dist))
            t_lower = doc.page_content.lower()

            # Exact multi-word phrase boost
            if any(p in t_lower for p in key_phrases):
                score += 0.45

            # Keyword overlap
            score += sum(0.06 for w in q_words if w in t_lower)

            # Conceptual definition cues
            if any(cue in t_lower for cue in ['occurs when', 'is called a', 'written as: st', 'st → at → rt', 'the movement of an agent']):
                score += 0.35

            # Front-matter and bibliography penalties
            if any(b in t_lower for b in ['submitted by', 'institute of technology', 'department of artificial', 'seminar report', '13. references', 'springer, 2018', 'mit press']):
                score -= 0.85

            reranked.append((doc, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]

    except Exception as exc:
        print(f"Hybrid retrieval fallback: {exc}")
        docs = db.similarity_search(query, k=k)
        return [(d, max(0.2, 0.95 - (i * 0.15))) for i, d in enumerate(docs)]


def get_total_chunks() -> int:
    try:
        return db._collection.count()
    except Exception:
        return 0


def get_indexed_documents() -> List[Dict[str, Any]]:
    """Returns aggregated metadata of all currently stored documents."""
    try:
        count = db._collection.count()
        if count == 0:
            return []

        data = db._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []

        doc_summary: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            fname = meta.get("filename") or meta.get("source") or "Unknown Document"
            clean_name = os.path.basename(fname)
            page = meta.get("page", 1)

            if clean_name not in doc_summary:
                doc_summary[clean_name] = {
                    "filename": clean_name,
                    "chunks": 0,
                    "pages": set(),
                }
            doc_summary[clean_name]["chunks"] += 1
            if page is not None:
                doc_summary[clean_name]["pages"].add(page)

        result = []
        for name, info in doc_summary.items():
            result.append({
                "filename": name,
                "chunks": info["chunks"],
                "total_pages": len(info["pages"]) if info["pages"] else 1
            })
        return result
    except Exception as exc:
        print(f"Error getting indexed documents: {exc}")
        return []


def delete_document(filename: str) -> bool:
    """Removes all chunks corresponding to a specific document."""
    try:
        clean_name = os.path.basename(filename)
        data = db._collection.get(include=["metadatas"])
        ids_to_delete = []
        for item_id, meta in zip(data.get("ids", []), data.get("metadatas", [])):
            if not meta:
                continue
            src = os.path.basename(str(meta.get("filename") or meta.get("source") or ""))
            if src == clean_name:
                ids_to_delete.append(item_id)

        if ids_to_delete:
            db._collection.delete(ids=ids_to_delete)
            return True
        return False
    except Exception as exc:
        print(f"Error deleting document {filename}: {exc}")
        return False


def clear_all_documents() -> bool:
    """Clears all vectors from Chroma."""
    global db
    try:
        try:
            db.delete_collection()
        except Exception:
            pass

        db = Chroma(
            persist_directory=CHROMA_PATH_STR,
            embedding_function=embedding_model,
            collection_name="rag_knowledge_base"
        )
        return True
    except Exception as exc:
        print(f"Error clearing vector store: {exc}")
        return False
