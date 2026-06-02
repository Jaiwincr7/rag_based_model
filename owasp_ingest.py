"""Build OWASP Chroma index from GitHub markdown (run locally; not on Render 512MB)."""

import gc
import os
import shutil

import requests
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from owasp_utils import is_junk_chunk

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/OWASP/Top10/master/2021/docs/en"
)

OWASP_MARKDOWN_FILES = [
    "index.md",
    "A00_2021_Introduction.md",
    "A01_2021-Broken_Access_Control.md",
    "A02_2021-Cryptographic_Failures.md",
    "A03_2021-Injection.md",
    "A04_2021-Insecure_Design.md",
    "A05_2021-Security_Misconfiguration.md",
    "A06_2021-Vulnerable_and_Outdated_Components.md",
    "A07_2021-Identification_and_Authentication_Failures.md",
    "A08_2021-Software_and_Data_Integrity_Failures.md",
    "A09_2021-Security_Logging_and_Monitoring_Failures.md",
    "A10_2021-Server-Side_Request_Forgery_(SSRF).md",
]

DB_PATH = "./chroma_db/owasp"


def is_useful_content(text: str) -> bool:
    return not is_junk_chunk(text)


def load_markdown_doc(filename: str) -> Document | None:
    url = f"{GITHUB_RAW_BASE}/{filename}"
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    text = resp.text
    if not is_useful_content(text):
        return None
    return Document(
        page_content=text,
        metadata={"source": f"OWASP/en/{filename}", "url": url},
    )


def build_owasp_db(force: bool = False) -> int:
    """
    Download OWASP Top 10 2021 (en) and write chroma_db/owasp.
    Processes one file at a time to limit peak RAM (for local ingest).
    """
    from rag_components import FastEmbedEmbeddings, owasp_embedding_model

    if force and os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH, ignore_errors=True)

    embeddings = FastEmbedEmbeddings(model_name=owasp_embedding_model)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    vectorstore = None
    total = 0

    for filename in OWASP_MARKDOWN_FILES:
        doc = load_markdown_doc(filename)
        if not doc:
            continue

        splits = splitter.split_documents([doc])
        splits = [s for s in splits if is_useful_content(s.page_content)]
        del doc

        if not splits:
            gc.collect()
            continue

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                collection_name="owasp",
                persist_directory=DB_PATH,
            )
        else:
            vectorstore.add_documents(splits)

        total += len(splits)
        del splits
        gc.collect()

    if vectorstore is None or total == 0:
        raise RuntimeError(
            "Could not build OWASP index. Check network and GitHub URLs."
        )

    return total
