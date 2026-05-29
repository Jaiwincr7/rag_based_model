import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint

model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
use_remote_llm = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"
use_remote_embeddings = os.getenv("USE_REMOTE_EMBEDDINGS", "true").lower() == "true"

_llm = None
_embeddings = None


def get_embeddings():
    """
    Remote embeddings via Hugging Face API — no local torch/sentence-transformers load.
    Required for Render free tier (512MB).
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if not use_remote_embeddings:
        raise RuntimeError(
            "Local embeddings are disabled for deployment. "
            "Set USE_REMOTE_EMBEDDINGS=true and HF_TOKEN on Render."
        )

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for remote embeddings.")

    _embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=token,
        model_name=embedding_model,
    )
    return _embeddings


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if not use_remote_llm:
        raise RuntimeError(
            "Local LLM is disabled for deployment. "
            "Set USE_REMOTE_LLM=true and HF_TOKEN on Render."
        )

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for remote LLM.")

    _llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=token,
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
