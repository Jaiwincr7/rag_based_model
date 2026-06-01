import os

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from rag_components import get_mitre_embeddings

CHROMA_PATH = "./chroma_db/mitre_attack_v5"
COLLECTION_NAME = "mitre_enterprise_attack_v5"

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    if not os.path.isdir(CHROMA_PATH):
        raise FileNotFoundError(
            f"MITRE vector DB not found at {CHROMA_PATH}. "
            "Run ingest_mitre.py locally and deploy chroma_db to Render."
        )

    _vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_mitre_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    return _vectorstore
