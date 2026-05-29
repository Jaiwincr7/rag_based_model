import os
from langchain_community.embeddings import HuggingFaceEmbeddings

_embeddings = None


def get_embeddings():
    """Single shared embedding model instance (avoids loading twice in RAM)."""
    global _embeddings
    if _embeddings is None:
        model = os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _embeddings = HuggingFaceEmbeddings(model_name=model)
    return _embeddings
