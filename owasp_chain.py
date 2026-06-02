import os

from rag_components import format_docs, hf_chat_complete
from owasp_store import retrieve_owasp_docs

SYSTEM_TEMPLATE = """You are a helpful cybersecurity assistant.
Use the following retrieved context to answer the question.
If the answer is not in the context, say "I don't know".
Keep the answer concise.

Context:
{context}
"""


def owasp_print(query: str) -> str:
    docs = retrieve_owasp_docs(query)
    context = format_docs(docs)

    if os.getenv("USE_EXTRACTIVE_ONLY", "false").lower() == "true":
        return context

    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)},
        {"role": "user", "content": query},
    ]
    try:
        return hf_chat_complete(messages)
    except RuntimeError as e:
        msg = str(e)
        if "model_not_supported" in msg or "not supported by any provider" in msg:
            return (
                "Generative LLM unavailable on HF; retrieved OWASP excerpts:\n\n"
                f"{context}"
            )
        raise
