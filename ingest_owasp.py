"""
Ingest OWASP Top 10 2021 from GitHub English markdown.
Run locally, then deploy ./chroma_db/owasp to Render:

  pip install requests sentence-transformers langchain-community chromadb langchain-text-splitters
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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "./chroma_db/owasp"


def is_useful_content(text: str) -> bool:
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
    print(f"  fetching {filename} ...")
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        print(f"  ⚠️  404: {filename}")
        return None
    resp.raise_for_status()
    text = resp.text
    if not is_useful_content(text):
        print(f"  ⚠️  skipped low-quality: {filename}")
        return None
    return Document(
        page_content=text,
        metadata={"source": f"OWASP/en/{filename}", "url": url},
    )


def main():
    print("📥 Loading OWASP Top 10 2021 (GitHub en/)...")
    docs: list[Document] = []
    for filename in OWASP_MARKDOWN_FILES:
        doc = load_markdown_doc(filename)
        if doc:
            docs.append(doc)

    if not docs:
        print("❌ No documents loaded. Check internet or filenames.")
        sys.exit(1)

    print(f"✅ Loaded {len(docs)} documents.")
    sample = docs[0].page_content[:180].replace("\n", " ")
    print(f"🔎 Sample: {sample}...")

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

    print(f"⏳ Embedding ({EMBEDDING_MODEL}) — first run downloads the model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print(f"🧹 Removed old DB at {DB_PATH}")
        except PermissionError:
            print("⚠️  Close apps using chroma_db, then rerun.")

    print("💾 Writing ChromaDB...")
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="owasp",
        persist_directory=DB_PATH,
    )
    print(f"🎉 Done. Deploy folder: {DB_PATH}")


if __name__ == "__main__":
    main()
