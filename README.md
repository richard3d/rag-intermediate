# RAG-Intermediate

A Retrieval-Augmented Generation (RAG) system that evolves the [rag-basic](https://github.com/richard3d/rag-basic) project by introducing two architectural improvements that were called out as future work: an LLM gateway (LiteLLM) and a proper vector store abstraction (LangChain's `PGVector`).

## What Changed from RAG-Basic

The basic version called Ollama directly from application code using `langchain_ollama`'s `ChatOllama` and `OllamaEmbeddings`, and managed all vector storage through raw `psycopg` SQL. This intermediate version addresses both of those trade-offs:

**LiteLLM gateway for all model calls** — a LiteLLM container now sits between the RAG service and Ollama. The RAG service talks to LiteLLM's OpenAI-compatible REST API using `ChatOpenAI` and `OpenAIEmbeddings` from `langchain_openai` instead of `ChatOllama` and `OllamaEmbeddings`. Both generative and embedding calls are routed through the gateway, so swapping any underlying model is a config change in LiteLLM, not a code change in the RAG service.

**`PGVector` store instead of raw SQL** — `embed.py` and `query.py` now use LangChain's `PGVector` from `langchain_postgres` for all vector operations. The hand-written DDL, raw `psycopg` inserts, and cosine similarity SQL are gone. `PGVector` manages the schema, batching, and similarity search internally.

## What It Does

Drop a text file into a watched directory and the system automatically chunks it, embeds it, and stores the vectors in a PostgreSQL database. Then ask questions against those documents via a REST API and get answers grounded in the ingested content.

```
Document → Chunker → Embedder (LiteLLM) → PGVector (pgvector DB)
                                               ↑
Question → Embedder (LiteLLM) → Similarity Search → LiteLLM → Ollama LLM → Answer
```

## Stack

| Layer | Technology |
|---|---|
| LLM | [LiteLLM](https://github.com/BerriAI/litellm) gateway → [Ollama](https://ollama.com) (`llama3.2`) |
| Embeddings | [LiteLLM](https://github.com/BerriAI/litellm) gateway → [Ollama](https://ollama.com) (`nomic-embed-text`) via `langchain_openai` |
| Vector Store | [LangChain PGVector](https://python.langchain.com/docs/integrations/vectorstores/pgvector/) (`langchain_postgres`) + [pgvector](https://github.com/pgvector/pgvector) |
| RAG Orchestration | [LangChain](https://python.langchain.com) (`langchain_openai`, `langchain_postgres`) |
| API Server | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) |
| File Watching | [Watchdog](https://github.com/gorakhargosh/watchdog) |
| Runtime | Python 3.14, [uv](https://github.com/astral-sh/uv) |
| Infrastructure | Docker Compose |

## Architecture

The RAG service (`src/`) has two concurrent responsibilities, run in separate threads from `main.py`:

**Ingest pipeline** — a Watchdog file observer monitors a directory for new files. When a file arrives:
1. `chunk.py` reads it asynchronously and splits it into overlapping text chunks using LangChain's `RecursiveCharacterTextSplitter`
2. `embed.py` calls the LiteLLM gateway via `OpenAIEmbeddings` to generate embeddings, then writes them to pgvector through `PGVector.aadd_texts()`

**Query pipeline** — a FastAPI server exposes a `POST /query-rag` endpoint. When a question arrives:
1. The question is embedded using `OpenAIEmbeddings` (via LiteLLM) and the top-3 most similar chunks are retrieved via `PGVector.similarity_search()`
2. Those chunks are injected into a prompt template and sent to `ChatOpenAI` pointed at the LiteLLM gateway (`http://litellm:4000/v1`)
3. LiteLLM routes the request to `llama3.2` on Ollama and the grounded answer is returned

## Project Structure

```
rag-intermediate/
├── src/
│   ├── main.py       # Entry point — starts API server and file observer concurrently
│   ├── server.py     # FastAPI app with /query-rag endpoint
│   ├── ingest.py     # Watchdog observer that triggers the embed pipeline on new files
│   ├── chunk.py      # Async streaming text chunker (LangChain splitter)
│   ├── embed.py      # Generates embeddings and writes to pgvector via PGVector
│   ├── query.py      # Retrieves relevant chunks via PGVector, calls LLM via LiteLLM
│   └── config.py     # Environment-based configuration
├── litellm/
│   └── config.yaml              # LiteLLM model routing config
├── pgvector/
│   └── init-scripts/init.sql   # Enables the pgvector extension on startup
├── docker-compose.yaml
├── Dockerfile
└── pyproject.toml
```

## Running It

**Prerequisites:** Docker and Docker Compose.

```bash
# Pull images, build the RAG container, and start everything
docker compose up --build
```

On first run, `ollama-init` pulls `llama3.2` and `nomic-embed-text` — this may take a few minutes depending on your connection.

Services:
- RAG API: `http://localhost:8000`
- LiteLLM gateway: `http://localhost:4000`
- Ollama: `http://localhost:11434`

## Usage

**Ingest a document** — copy any `.txt` file into `./docs/` (created automatically on first run):

```bash
echo "The capital of France is Paris." > ./docs/facts.txt
```

Optionally, copy the example text on Scottish History from `./resources/` into `./docs/`

The file observer picks it up immediately and logs the processing steps.

**Query the RAG system:**

```bash
curl -X POST http://localhost:8000/query-rag \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

```json
{"response": "The capital of France is Paris."}
```

Note: If you are using the provided example resource, try prompting `Tell me about Henry Mackenzie`

You can also enter prompts via the Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs)

## GPU Acceleration

CPU inference works out of the box. For GPU support on Linux with an NVIDIA card, uncomment the `deploy` block in `docker-compose.yaml` under the `ollama` service and install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Key Concepts Explored

- **LLM gateway pattern** — LiteLLM exposes an OpenAI-compatible API so the RAG service can use `ChatOpenAI` regardless of what model or provider is running underneath; switching models is a gateway config change, not a code change
- **Provider portability** — because the RAG service speaks OpenAI-compatible REST, you could point LiteLLM at a cloud provider (e.g. Anthropic, OpenAI, Bedrock) by adding an entry to `litellm/config.yaml` with no changes to application code
- **Vector store abstraction** — `PGVector` from `langchain_postgres` handles schema management, upserts, and similarity search so application code works at the level of documents and queries rather than SQL strings
- **Rate limiting embedding calls** — While Litellm currently is configured to use Ollama embeddings, a cloud-based provider can be swapped in now. In that case we should be conscious of API calls especially if a ton of documents got dropped into the system all at once. Since this is a small project `InMemoryRateLimiter` from `langchain_core` is used, however in a production-grade system, I would opt for actual request-queueing with associated infrastructure; resilience and scalability being the obvious improvements.

## Design Trade-offs & Potential Improvements

- **Hybrid search** - I've been looking at ParadeDB recently due to recommendation from a colleague [@zoe-codez](https://github.com/zoe-codez). After researching further I think this would be an obvious next step from just using a vector store since it supports full-text search. Vector similarity search captures semantic meaning well, but struggles with exact-match lookups like product IDs or serial numbers. Embeddings smooth over lexical differences, and rare identifiers may have unreliable representations. Hybrid search addresses this by combining vector similarity with BM25 full-text scoring (typically fused via Reciprocal Rank Fusion), providing semantic recall AND keyword precision in a single query.

- **Multimodal RAG** - Up to this point, the input files for ingestion have been limited. The application still only supports UTF-8 text files. Parsing common types like `csv` and `pdf` is a natural evolutionary step. But, we can take this idea even futher; building a multimodal RAG system would allow us to ingest and embed things like images and charts and not just text.

- **OpenAI API compatibility** - The `POST /query-rag` endpoint was hastily cobbled together just to get an interface to interact with the RAG pipeline. But, it isn't drop-in compatible with the broader AI ecosystem. Implementing it as `/v1/chat/completions` with an OpenAI-compatible request/response shape would allow tools like AnythingLLM, Open WebUI, or any OpenAI-compatible client to sit in front of it without modification — treating this service like any other LLM endpoint.

