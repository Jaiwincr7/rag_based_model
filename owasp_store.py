import os

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from rag_components import get_owasp_embeddings
from owasp_utils import filter_good_docs, is_junk_chunk

_store = None
_rebuild_done = False
OWASP_DB_PATH = "./chroma_db/owasp"


def _reset_store():
    global _store
    _store = None


def get_owasp_store():
    global _store
    if _store is not None:
        return _store

    if not os.path.isdir(OWASP_DB_PATH):
        _try_build_db()

    _store = Chroma(
        collection_name="owasp",
        persist_directory=OWASP_DB_PATH,
        embedding_function=get_owasp_embeddings(),
    )
    return _store


def _try_build_db():
    if os.getenv("AUTO_BUILD_OWASP_DB", "true").lower() != "true":
        raise FileNotFoundError(
            f"Missing {OWASP_DB_PATH}. Run: python ingest_owasp.py"
        )
    from owasp_ingest import build_owasp_db

    build_owasp_db(force=True)


def get_owasp_retriever():
    return get_owasp_store().as_retriever(search_kwargs={"k": 8})


def retrieve_owasp_docs(query: str, k: int = 12, min_good: int = 2) -> list:
    global _rebuild_done

    store = get_owasp_store()
    candidates = store.similarity_search(query, k=k)
    good = filter_good_docs(candidates)[:min_good]

    if good:
        return good

    if (
        candidates
        and all(is_junk_chunk(d.page_content) for d in candidates)
        and not _rebuild_done
        and os.getenv("AUTO_BUILD_OWASP_DB", "true").lower() == "true"
    ):
        _rebuild_done = True
        _reset_store()
        from owasp_ingest import build_owasp_db

        build_owasp_db(force=True)
        _reset_store()
        store = get_owasp_store()
        candidates = store.similarity_search(query, k=k)
        good = filter_good_docs(candidates)[:min_good]
        if good:
            return good

    if candidates and all(is_junk_chunk(d.page_content) for d in candidates):
        raise FileNotFoundError(
            "OWASP index still invalid after rebuild. "
            "Run locally: python ingest_owasp.py  then push chroma_db/owasp."
        )

    return good
