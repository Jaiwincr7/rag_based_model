import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_components import format_docs, get_llm
from owasp_store import get_owasp_retriever

template = """You are a helpful cybersecurity assistant.
Use the following pieces of retrieved context to answer the question.
If the answer is not in the context, just say "I don't know".
Keep the answer concise and professional.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{question}"),
])


def owasp_print(query):
    retriever = get_owasp_retriever()

    # No LLM RAM — returns retrieved chunks only (set on Render if still OOM)
    if os.getenv("USE_EXTRACTIVE_ONLY", "false").lower() == "true":
        docs = retriever.invoke(query)
        return format_docs(docs)

    llm = get_llm()
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)
