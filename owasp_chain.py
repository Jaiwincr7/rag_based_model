import os
import re

from rag_components import format_docs, hf_chat_complete
from owasp_store import get_owasp_store, retrieve_owasp_docs

SYSTEM_TEMPLATE = """You are a helpful cybersecurity assistant.
Use the following retrieved context to answer the question.
If the answer is not in the context, say "I don't know".
Keep the answer concise.

Context:
{context}
"""

# Official OWASP Top 10:2021 categories (matches ingest_owasp.py filenames)
OWASP_TOP10_2021 = [
    ("A01_2021-Broken_Access_Control.md", "A01:2021 – Broken Access Control"),
    ("A02_2021-Cryptographic_Failures.md", "A02:2021 – Cryptographic Failures"),
    ("A03_2021-Injection.md", "A03:2021 – Injection"),
    ("A04_2021-Insecure_Design.md", "A04:2021 – Insecure Design"),
    ("A05_2021-Security_Misconfiguration.md", "A05:2021 – Security Misconfiguration"),
    ("A06_2021-Vulnerable_and_Outdated_Components.md", "A06:2021 – Vulnerable and Outdated Components"),
    ("A07_2021-Identification_and_Authentication_Failures.md", "A07:2021 – Identification and Authentication Failures"),
    ("A08_2021-Software_and_Data_Integrity_Failures.md", "A08:2021 – Software and Data Integrity Failures"),
    ("A09_2021-Security_Logging_and_Monitoring_Failures.md", "A09:2021 – Security Logging and Monitoring Failures"),
    ("A10_2021-Server-Side_Request_Forgery_(SSRF).md", "A10:2021 – Server-Side Request Forgery (SSRF)"),
]


def _use_extractive_mode() -> bool:
    if os.getenv("USE_EXTRACTIVE_ONLY", "false").lower() == "true":
        return True
    return os.getenv("USE_REMOTE_LLM", "true").lower() == "false"


def _is_list_top10_query(query: str) -> bool:
    q = query.lower().replace("-", " ")
    if "owasp" not in q and "top 10" not in q and "top10" not in q:
        return False
    return any(
        phrase in q
        for phrase in ("list", "what are", "name all", "enumerate", "show all", "top ten")
    ) or ("top 10" in q and len(q.split()) <= 6)


def _summary_from_doc(doc) -> str:
    text = doc.page_content
    match = re.search(
        r"Description:\s*(.+?)(?:\n\s*(?:#{1,3}\s|Tactics:)|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        summary = match.group(1).strip()
    else:
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip()
            and not ln.startswith("#")
            and not ln.startswith("!")
            and "{" not in ln[:20]
        ]
        summary = lines[0] if lines else text.strip()

    summary = re.sub(r"\s+", " ", summary)
    if len(summary) > 240:
        summary = summary[:240].rsplit(" ", 1)[0] + "..."
    return summary


def list_owasp_top10() -> str:
    """Return all 10 categories with a one-line description each."""
    store = get_owasp_store()
    lines = ["## OWASP Top 10:2021 — full list\n"]

    for i, (filename, label) in enumerate(OWASP_TOP10_2021, start=1):
        source = f"OWASP/en/{filename}"
        hits = store.similarity_search(
            label,
            k=5,
            filter={"source": source},
        )
        if not hits:
            hits = [
                d
                for d in store.similarity_search(label, k=10)
                if d.metadata.get("source") == source
            ]

        if hits:
            summary = _summary_from_doc(hits[0])
        else:
            summary = "Description not loaded in index — re-run ingest_owasp.py."

        lines.append(f"{i}. **{label}**\n   {summary}\n")

    return "\n".join(lines)


def owasp_print(query: str) -> str:
    if _is_list_top10_query(query):
        return list_owasp_top10()

    docs = retrieve_owasp_docs(query)
    context = format_docs(docs)

    if _use_extractive_mode():
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
