import os

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from rag_components import get_owasp_embeddings

_retriever = None
OWASP_DB_PATH = "./chroma_db/owasp"


def get_owasp_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    if not os.path.isdir(OWASP_DB_PATH):
        raise FileNotFoundError(
            f"OWASP vector DB not found at {OWASP_DB_PATH}. "
            "Run ingest_owasp.py locally and deploy the chroma_db/owasp folder to Render."
        )

    owasp_store = Chroma(
        collection_name="owasp",
        persist_directory=OWASP_DB_PATH,
        embedding_function=get_owasp_embeddings(),
    )
    _retriever = owasp_store.as_retriever(search_kwargs={"k": 2})
    return _retriever
