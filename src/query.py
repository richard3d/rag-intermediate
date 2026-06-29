from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import (
    DB_CONNECTION_STR,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    LITELLM_MODEL,
)

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based strictly on the provided context.
If you do not know the answer or if it's not explicitly in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

def retrieve_chunks(question: str, k: int = 3) -> list[str]:
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    store = PGVector(
        embeddings=embedder,
        collection_name="documents",
        connection=DB_CONNECTION_STR,
    )
    docs = store.similarity_search(question, k=k)
    return [doc.page_content for doc in docs]

def query_rag(question: str) -> str:
    chunks = retrieve_chunks(question)
    context = "\n\n".join(chunks)

    llm = ChatOpenAI(
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_API_KEY,
        model=LITELLM_MODEL,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
