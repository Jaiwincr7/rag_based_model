import os

from rag_components import format_docs, hf_chat_complete
from owasp_store import get_owasp_retriever

SYSTEM_TEMPLATE = """You are a helpful cybersecurity assistant.
Use the following retrieved context to answer the question.
If the answer is not in the context, say "I don't know".
Keep the answer concise.

Context:
{context}
"""


def owasp_print(query: str) -> str:
    retriever = get_owasp_retriever()

    # No LLM RAM — returns retrieved chunks only (set on Render if still OOM)
    if os.getenv("USE_EXTRACTIVE_ONLY", "false").lower() == "true":
        docs = retriever.invoke(query)
        return format_docs(docs)

    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)},
        {"role": "user", "content": query},
    ]
    try:
        return hf_chat_complete(messages)
    except RuntimeError as e:
        # Keep API usable when HF Router model/provider access is unavailable.
        msg = str(e)
        if "model_not_supported" in msg or "not supported by any provider" in msg:
            return (
                "Note: generative LLM is unavailable on your HF account; "
                "showing retrieved OWASP excerpts:\n\n"
                f"{context}"
            )
        raise
