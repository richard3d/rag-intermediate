import os

FILE_OBSERVER_DIRECTORY = os.environ.get("FILE_OBSERVER_DIRECTORY", "/app/docs")
# Ensure the connection string uses the psycopg SQLAlchemy dialect, this gives us access to async operations
DB_CONNECTION_STR = os.environ.get(
    "DB_CONNECTION_STR",
    "postgresql+psycopg://postgres:password@db:5432/rag-intermediate",
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
# Defaults to 768 to match dimensions for the default model nomic-embed-text
EMBEDDING_MODEL_VEC_DIMENSION = os.environ.get("EMBEDDING_MODEL_VEC_DIMENSION", 768)
# Option to rate limit embedding requests to LiteLLM, useful if using a cloud-based provider
EMBEDDING_REQUESTS_PER_SECOND = os.environ.get("EMBEDDING_REQUESTS_PER_SECOND", 25)
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://litellm:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-local")
LITELLM_MODEL = os.environ.get("LITELLM_MODEL", "ollama/llama3.2")
