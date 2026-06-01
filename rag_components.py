import os
from pathlib import Path

from hf_legacy import LegacyHFEmbeddings, LegacyHFLLM

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
_DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
owasp_embedding_model = os.getenv("OWASP_EMBEDDING_MODEL", _DEFAULT_EMBEDDING)
mitre_embedding_model = os.getenv("MITRE_EMBEDDING_MODEL", _DEFAULT_EMBEDDING)
use_remote_llm = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"
use_remote_embeddings = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"
# Legacy api-inference.huggingface.co — works with normal read tokens (no Inference Providers).
use_hf_legacy_api = os.getenv("USE_HF_LEGACY_API", "true").lower() == "true"
hf_timeout = int(os.getenv("HF_TIMEOUT_SEC", "90"))

_llm = None
_owasp_embeddings = None
_mitre_embeddings = None


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
        "Missing Hugging Face token. In Render → Environment add HF_TOKEN=hf_... "
        "then Save and redeploy."
    )


def _make_remote_embeddings(model: str):
    if not use_remote_embeddings:
        raise RuntimeError("Remote embeddings required. Set USE_REMOTE_EMBEDDINGS=true.")

    token = get_hf_token()
    if use_hf_legacy_api:
        return LegacyHFEmbeddings(model_id=model, api_key=token, timeout=hf_timeout)

    # Fallback: LangChain HF router (needs Inference Providers permission on token).
    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        return HuggingFaceEndpointEmbeddings(
            model=model, huggingfacehub_api_token=token
        )
    except ImportError:
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

        return HuggingFaceInferenceAPIEmbeddings(api_key=token, model_name=model)


def get_owasp_embeddings():
    global _owasp_embeddings
    if _owasp_embeddings is None:
        _owasp_embeddings = _make_remote_embeddings(owasp_embedding_model)
    return _owasp_embeddings


def get_mitre_embeddings():
    global _mitre_embeddings
    if _mitre_embeddings is None:
        _mitre_embeddings = _make_remote_embeddings(mitre_embedding_model)
    return _mitre_embeddings


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if not use_remote_llm:
        raise RuntimeError("Local LLM is disabled. Set USE_REMOTE_LLM=true.")

    token = get_hf_token()
    if use_hf_legacy_api:
        _llm = LegacyHFLLM(
            model_id=model_name,
            api_key=token,
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
            timeout=hf_timeout,
        )
    else:
        from langchain_huggingface import HuggingFaceEndpoint

        _llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=token,
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
            temperature=0.2,
            task="text-generation",
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
