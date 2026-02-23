import bs4
import shutil
import os
import sys
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma 

# 1. CORRECT URLs
urls = [
    "https://owasp.org/Top10/A00_2021_Introduction/",
    "https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
    "https://owasp.org/Top10/A02_2021-Cryptographic_Failures/",
    "https://owasp.org/Top10/A03_2021-Injection/",
    "https://owasp.org/Top10/A04_2021-Insecure_Design/",
    "https://owasp.org/Top10/A05_2021-Security_Misconfiguration/",
    "https://owasp.org/Top10/A06_2021-Vulnerable_and_Outdated_Components/",
    "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/",
    "https://owasp.org/Top10/A08_2021-Software_and_Data_Integrity_Failures/",
    "https://owasp.org/Top10/A09_2021-Security_Logging_and_Monitoring_Failures/",
    "https://owasp.org/Top10/A10_2021-Server-Side_Request_Forgery_%28SSRF%29/",
]

print("📥 Downloading pages...")

# 2. NO FILTER (Safest Approach)
# We removed bs_kwargs to ensure we get ALL text.
loader = WebBaseLoader(
    web_path=urls,
    header_template={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
)
docs = loader.load()

# DEBUG: Check if we actually have text now
print(f"✅ Found {len(docs)} documents.")
if len(docs) > 0:
    print(f"🔎 Sample content from Doc 1: {docs[1].page_content[:200]}...") 

# 3. Add Metadata
for d in docs:
    d.metadata["source"] = "OWASP"

# 4. Split Text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
splits = splitter.split_documents(docs)

if len(splits) == 0:
    print("❌ ERROR: Documents are still empty! The website might require JavaScript.")
    sys.exit()

print(f"✅ Created {len(splits)} text chunks.")

# 5. Embeddings
print("⏳ Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="ibm-granite/granite-embedding-107m-multilingual"
)

# 6. SAVE TO DISK
db_path = "./chroma_db/owasp"

# CLEANUP: Remove old DB to prevent corruption
if os.path.exists(db_path):
    try:
        shutil.rmtree(db_path)
        print(f"🧹 Removed old database at {db_path}")
    except PermissionError:
        print("⚠️  Warning: Could not delete old DB folder. Please close any running terminals using it.")

print("💾 Saving to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="owasp",
    persist_directory=db_path
)

print(f"🎉 SUCCESS: Database created at {db_path}")