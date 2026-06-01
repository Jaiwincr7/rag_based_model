"""
Hugging Face Serverless Inference API (api-inference.huggingface.co).

Works with standard read tokens. Does NOT use router.huggingface.co (Inference Providers).
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM

HF_INFERENCE_BASE = "https://api-inference.huggingface.co/models"


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _pool_embedding(raw) -> List[float]:
    if raw is None:
        raise ValueError("Empty embedding response")
    if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]
    arr = np.array(raw, dtype=float)
    if arr.ndim == 1:
        return arr.tolist()
    return arr.mean(axis=0).tolist()


class LegacyHFEmbeddings(Embeddings):
    def __init__(self, model_id: str, api_key: str, timeout: int = 90):
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.url = f"{HF_INFERENCE_BASE}/{model_id}"

    def _embed_one(self, text: str) -> List[float]:
        headers = _auth_headers(self.api_key)
        payload = {"inputs": text, "options": {"wait_for_model": True}}

        for attempt in range(3):
            resp = requests.post(
                self.url, headers=headers, json=payload, timeout=self.timeout
            )
            if resp.status_code == 503:
                time.sleep(min(2**attempt, 8))
                continue
            resp.raise_for_status()
            return _pool_embedding(resp.json())

        resp.raise_for_status()
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


class LegacyHFLLM(LLM):
    model_id: str
    api_key: str
    max_new_tokens: int = 80
    timeout: int = 90

    @property
    def _llm_type(self) -> str:
        return "legacy_hf_inference"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        headers = _auth_headers(self.api_key)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": 0.2,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        for attempt in range(3):
            resp = requests.post(
                f"{HF_INFERENCE_BASE}/{self.model_id}",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 503:
                time.sleep(min(2**attempt, 8))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        else:
            resp.raise_for_status()

        if isinstance(data, list) and data:
            item = data[0]
            if isinstance(item, dict):
                return item.get("generated_text", str(item))
        if isinstance(data, dict):
            return data.get("generated_text", str(data))
        return str(data)
