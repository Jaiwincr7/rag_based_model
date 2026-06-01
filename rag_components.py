import os
from pathlib import Path
from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
owasp_embedding_model = os.getenv("OWASP_EMBEDDING_MODEL", _DEFAULT_EMBEDDING)
mitre_embedding_model = os.getenv("MITRE_EMBEDDING_MODEL", _DEFAULT_EMBEDDING)

# fastembed = local ONNX (~80MB). hf_router = needs Inference Providers on token.
embedding_backend = os.getenv("EMBEDDING_BACKEND", "fastembed").lower()
use_remote_llm = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"
hf_timeout = int(os.getenv("HF_TIMEOUT_SEC", "90"))
hf_router_base = os.getenv("HF_ROUTER_BASE_URL", "https://router.huggingface.co/v1")

_llm = None
_embedding_cache: dict[str, object] = {}


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings — no Hugging Face Inference API required."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()


class HFRouterLLM(LLM):
    """Hugging Face Inference Providers router (OpenAI-compatible chat API)."""

    model_id: str
    api_key: str
    max_tokens: int = 80
    timeout: int = 90
    router_base: str = "https://router.huggingface.co/v1"

    @property
    def _llm_type(self) -> str:
        return "hf_router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        url = f"{self.router_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }

        resp = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )

        if resp.status_code == 401:
            raise RuntimeError(
                "HF router 401: invalid or missing token. Use a fine-grained HF token "
                "with 'Inference Providers' permission, or set USE_EXTRACTIVE_ONLY=true."
            )

        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def hf_token_configured() -> bool:
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip() and value.strip() != "your_huggingface_token_here":
            return True
    return False


def _clean_token(value: str) -> str:
    token = value.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        token = token[1:-1].strip()
    return token


def get_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip() and value.strip() != "your_huggingface_token_here":
            return _clean_token(value)
    raise RuntimeError(
        "Missing Hugging Face token. Create one at huggingface.co/settings/tokens "
        "with 'Inference Providers' permission, set HF_TOKEN on Render, redeploy."
    )


def _build_embeddings(model: str):
    if embedding_backend == "fastembed":
        return FastEmbedEmbeddings(model_name=model)

    if embedding_backend == "hf_router":
        token = get_hf_token()
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            return HuggingFaceEndpointEmbeddings(
                model=model, huggingfacehub_api_token=token
            )
        except ImportError:
            from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

            return HuggingFaceInferenceAPIEmbeddings(api_key=token, model_name=model)

    raise ValueError(
        f"Unknown EMBEDDING_BACKEND={embedding_backend}. Use 'fastembed' or 'hf_router'."
    )


def get_owasp_embeddings():
    if owasp_embedding_model not in _embedding_cache:
        _embedding_cache[owasp_embedding_model] = _build_embeddings(owasp_embedding_model)
    return _embedding_cache[owasp_embedding_model]


def get_mitre_embeddings():
    if mitre_embedding_model not in _embedding_cache:
        _embedding_cache[mitre_embedding_model] = _build_embeddings(mitre_embedding_model)
    return _embedding_cache[mitre_embedding_model]


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if not use_remote_llm:
        raise RuntimeError("Local LLM is disabled. Set USE_REMOTE_LLM=true.")

    _llm = HFRouterLLM(
        model_id=model_name,
        api_key=get_hf_token(),
        max_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
        timeout=hf_timeout,
        router_base=hf_router_base,
    )
    return _llm


def format_docs(docs):
    formatted = []
    for d in docs:
        formatted.append(
            f"[SOURCE: {d.metadata.get('source')}\n"
            f"{d.page_content.strip()[:500]}"
        )
    return "\n\n".join(formatted)
