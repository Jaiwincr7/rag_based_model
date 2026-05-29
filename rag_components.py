import os
from pathlib import Path

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint

# Load .env locally (Render uses dashboard Environment variables instead).
try:
    from dotenv import load_dotenv

    # Never override variables already set by Render (or the shell).
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
use_remote_llm = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"
use_remote_embeddings = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"

_llm = None
_embeddings = None


def hf_token_configured() -> bool:
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip() and value.strip() != "your_huggingface_token_here":
            return True
    return False


def get_hf_token() -> str:
    """Read Hugging Face token from common environment variable names."""
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(key)
        if value and value.strip() and value.strip() != "your_huggingface_token_here":
            return value.strip()
    raise RuntimeError(
        "Missing Hugging Face token. In Render → Environment add HF_TOKEN=hf_... "
        "then Save and redeploy."
    )


def get_embeddings():
    """
    Remote embeddings via Hugging Face API — no local torch/sentence-transformers load.
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if not use_remote_embeddings:
        raise RuntimeError(
            "Local embeddings are disabled. Set USE_REMOTE_EMBEDDINGS=true."
        )

    _embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=get_hf_token(),
        model_name=embedding_model,
    )
    return _embeddings


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
        top_p=0.9,
        repetition_penalty=1.1,
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
