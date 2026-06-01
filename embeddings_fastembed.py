"""Lightweight local embeddings (no Hugging Face Inference API required)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()
