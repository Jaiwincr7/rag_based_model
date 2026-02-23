from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="ibm-granite/granite-embedding-107m-multilingual"
)

owasp_store = Chroma(
    collection_name="owasp",
    persist_directory="./chroma_db/owasp",
    embedding_function=embeddings
)

owasp_retriever = owasp_store.as_retriever(search_kwargs={"k": 5})
