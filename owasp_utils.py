"""Helpers to detect and filter bad OWASP chunks (e.g. redirect HTML)."""


def is_junk_chunk(text: str) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 200:
        return True
    lower = cleaned.lower()
    if "redirecting to owasp top 10" in lower:
        return True
    if lower.count("redirecting") >= 2 and len(cleaned) < 1500:
        return True
    return False


def filter_good_docs(docs: list) -> list:
    return [d for d in docs if not is_junk_chunk(d.page_content)]
