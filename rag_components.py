import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

from embeddings_fastembed import FastEmbedEmbeddings

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


def hf_token_configured() -> bool:
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip() and value.strip() != "your_huggingface_token_here":
            return True
    return False


def get_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip() and value.strip() != "your_huggingface_token_here":
            return value.strip()
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

    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        model=model_name,
        api_key=get_hf_token(),
        base_url=hf_router_base,
        max_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
        temperature=0.2,
        timeout=hf_timeout,
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
