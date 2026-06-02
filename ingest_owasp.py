"""
Ingest OWASP Top 10 2021 from GitHub markdown (stable; owasp.org often returns redirect HTML).
Run locally, then deploy ./chroma_db/owasp to Render.

  python ingest_owasp.py
"""

import os
import shutil
import sys

import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/OWASP/Top10/master/2021/docs"
)

# Filenames match OWASP/Top10 repo (2021/docs/)
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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./chroma_db/owasp"


def is_useful_content(text: str) -> bool:
    """Drop redirect stubs and near-empty pages."""
    cleaned = text.strip()
    if len(cleaned) < 200:
        return False
    lower = cleaned.lower()
    if "redirecting to owasp top 10" in lower and len(cleaned) < 800:
        return False
    if lower.count("redirecting") >= 2 and len(cleaned) < 1000:
        return False
    return True


def load_markdown_doc(filename: str) -> Document | None:
    url = f"{GITHUB_RAW_BASE}/{filename}"
    print(f"  fetching {url}")
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        print(f"  ⚠️  not found: {filename}")
        return None
    resp.raise_for_status()
    text = resp.text
    if not is_useful_content(text):
        print(f"  ⚠️  skipped low-quality content: {filename}")
        return None
    return Document(
        page_content=text,
        metadata={"source": f"OWASP/{filename}", "url": url},
    )


def main():
    print("📥 Loading OWASP Top 10 2021 from GitHub markdown...")
    docs: list[Document] = []
    for filename in OWASP_MARKDOWN_FILES:
        doc = load_markdown_doc(filename)
        if doc:
            docs.append(doc)

    if not docs:
        print("❌ No documents loaded. Check network or filenames.")
        sys.exit(1)

    print(f"✅ Loaded {len(docs)} documents.")
    print(f"🔎 Sample: {docs[0].page_content[:200].replace(chr(10), ' ')}...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    splits = splitter.split_documents(docs)
    splits = [s for s in splits if is_useful_content(s.page_content)]

    if not splits:
        print("❌ No chunks after splitting.")
        sys.exit(1)

    print(f"✅ Created {len(splits)} chunks.")

    print(f"⏳ Embedding with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"🧹 Removed old database at {DB_PATH}")
        except PermissionError:
            print("⚠️  Close apps using chroma_db, then rerun.")

    print("💾 Saving to ChromaDB...")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="owasp",
        persist_directory=DB_PATH,
    )
    print(f"🎉 SUCCESS: {DB_PATH} ready. Commit and deploy chroma_db/owasp to Render.")


if __name__ == "__main__":
    main()
