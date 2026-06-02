"""Build OWASP Chroma index from GitHub markdown (shared by CLI + runtime auto-rebuild)."""

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
    Uses fastembed — same as production on Render.
    Returns number of chunks stored.
    """
    from rag_components import FastEmbedEmbeddings, owasp_embedding_model

    if force and os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH, ignore_errors=True)

    docs: list[Document] = []
    for filename in OWASP_MARKDOWN_FILES:
        doc = load_markdown_doc(filename)
        if doc:
            docs.append(doc)

    if not docs:
        raise RuntimeError(
            "Could not download OWASP markdown from GitHub. Check network."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    splits = splitter.split_documents(docs)
    splits = [s for s in splits if is_useful_content(s.page_content)]

    if not splits:
        raise RuntimeError("No valid OWASP chunks after filtering.")

    embeddings = FastEmbedEmbeddings(model_name=owasp_embedding_model)
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="owasp",
        persist_directory=DB_PATH,
    )
    return len(splits)
    import sys

from owasp_ingest import build_owasp_db


def main():
    print("Building OWASP Chroma DB from GitHub (2021/docs/en)...")
    try:
        n = build_owasp_db(force=True)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
    print(f"🎉 SUCCESS: {n} chunks in ./chroma_db/owasp")
    print("Commit chroma_db/owasp to git OR redeploy Render (build runs ingest).")


if __name__ == "__main__":
    main()

