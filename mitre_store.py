from langchain_community.vectorstores import Chroma
from rag_components import get_embeddings

CHROMA_PATH = "./chroma_db/mitre_attack_v5"
COLLECTION_NAME = "mitre_enterprise_attack_v5"

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    _vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    return _vectorstore
