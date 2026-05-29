from langchain_community.vectorstores import Chroma
from rag_components import get_embeddings

_retriever = None


def get_owasp_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    owasp_store = Chroma(
        collection_name="owasp",
        persist_directory="./chroma_db/owasp",
        embedding_function=get_embeddings(),
    )
    _retriever = owasp_store.as_retriever(search_kwargs={"k": 2})
    return _retriever
