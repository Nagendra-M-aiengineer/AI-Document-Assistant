import hashlib
import logging
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class FastHashingEmbeddings(Embeddings):
    """
    Deterministic zero-dependency fallback embedding model with cosine-comparable dense representations.
    Used if heavy neural weights cannot be loaded.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().strip().split()
        if not words:
            return vec.tolist()

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign

        # Add n-gram hashing
        for i in range(len(text) - 2):
            trigram = text[i:i+3].encode("utf-8")
            h = int(hashlib.sha256(trigram).hexdigest(), 16)
            idx = h % self.dimension
            vec[idx] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


embedding_model = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    # Use locally cached or fast-loading sentence-transformers
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    logger.info("Initialized sentence-transformers/all-MiniLM-L6-v2 embedding model.")
except Exception as e:
    logger.warning(f"Could not load HuggingFaceEmbeddings ({e}), falling back to FastHashingEmbeddings.")
    embedding_model = FastHashingEmbeddings()
