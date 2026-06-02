import os

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from rag_components import get_owasp_embeddings
from owasp_utils import filter_good_docs, is_junk_chunk

_store = None
OWASP_DB_PATH = "./chroma_db/owasp"


def get_owasp_store():
    global _store
    if _store is not None:
        return _store

    if not os.path.isdir(OWASP_DB_PATH):
        raise FileNotFoundError(
            f"OWASP vector DB not found at {OWASP_DB_PATH}. "
            "Run: python ingest_owasp.py  then deploy chroma_db/owasp to Render."
        )

    _store = Chroma(
        collection_name="owasp",
        persist_directory=OWASP_DB_PATH,
        embedding_function=get_owasp_embeddings(),
    )
    return _store


def get_owasp_retriever():
    return get_owasp_store().as_retriever(search_kwargs={"k": 8})


def retrieve_owasp_docs(query: str, k: int = 12, min_good: int = 2) -> list:
    """Retrieve and drop redirect/junk chunks from old or bad indexes."""
    store = get_owasp_store()
    candidates = store.similarity_search(query, k=k)
    good = filter_good_docs(candidates)[:min_good]

    if good:
        return good

    # Old DB may only have redirect pages — tell user to re-ingest.
    if candidates and all(is_junk_chunk(d.page_content) for d in candidates):
        raise FileNotFoundError(
            "OWASP index contains only redirect/stub pages. "
            "On your machine run: python ingest_owasp.py "
            "then commit and redeploy the chroma_db/owasp folder."
        )

    return good
