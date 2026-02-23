from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_components import format_docs, llm 
from owasp_store import owasp_retriever

# 1. Use ChatPromptTemplate (Better for Chat/Instruct models)
# This creates a clear boundary between instructions and user input.
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
    print(f"DEBUG: Asking Chain: {query}")
    
    # 2. Build the Chain
    chain = (
        {
            "context": owasp_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 3. Invoke
    response = chain.invoke(query)
    return response