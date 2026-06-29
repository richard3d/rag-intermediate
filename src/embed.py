from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from chunk import chunk_text_file
from config import DB_CONNECTION_STR, OLLAMA_BASE_URL, EMBEDDING_MODEL

BATCH_SIZE = 50

async def embed_text_file(file_path):
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    engine = create_async_engine(DB_CONNECTION_STR)
    store = PGVector(
        embeddings=embedder,
        collection_name="documents",
        connection=engine,
    )

    # Ensure tables/collection exist, then clear stale data for this file
    await store.acreate_collection()
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'file_path' = :fp"),
            {"fp": file_path},
        )

    batch: list[str] = []
    async for chunk in chunk_text_file(file_path):
        batch.append(chunk)
        if len(batch) >= BATCH_SIZE:
            await store.aadd_texts(texts=batch, metadatas=[{"file_path": file_path}] * len(batch))
            batch.clear()
    if batch:
        await store.aadd_texts(texts=batch, metadatas=[{"file_path": file_path}] * len(batch))
