import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# Prefer new packages; fall back to langchain_community if not installed on Render.
try:
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

    _USE_LANGCHAIN_HF = True
except ImportError:
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    from langchain_community.llms import HuggingFaceEndpoint

    HuggingFaceEndpointEmbeddings = HuggingFaceInferenceAPIEmbeddings  # type: ignore
    _USE_LANGCHAIN_HF = False

model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# Use models supported by HF Inference API (read + inference token).
_DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
owasp_embedding_model = os.getenv("OWASP_EMBEDDING_MODEL", _DEFAULT_EMBEDDING)
mitre_embedding_model = os.getenv("MITRE_EMBEDDING_MODEL", _DEFAULT_EMBEDDING)
use_remote_llm = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"
use_remote_embeddings = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"

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
    if _USE_LANGCHAIN_HF:
        return HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=token,
        )
    return HuggingFaceEndpointEmbeddings(api_key=token, model_name=model)


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

    _llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=get_hf_token(),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
        temperature=0.2,
        task="text-generation",
        timeout=int(os.getenv("HF_TIMEOUT_SEC", "60")),
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
