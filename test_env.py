import os
from dotenv import load_dotenv

load_dotenv(override=True)

# neo4j configs
NEO4J_URI = os.environ.get('NEO4J_URI', "bolt://localhost:7687") 
NEO4J_USER = os.environ.get('NEO4J_USER', "neo4j")
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', "password")

# LLM configs
LLM_API_KEY = os.environ.get('LLM_API_KEY', "dummy")
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', "http://localhost:11434/v1")
LLM_MODEL = os.environ.get('LLM_MODEL' , "qwen3:8b")

# Embedder configs
EMBEDDER_API_KEY = os.environ.get('EMBEDDER_API_KEY', "dummy")
EMBEDDER_BASE_URL = os.environ.get("EMBEDDER_BASE_URL", "http://localhost:11434/v1")
EMBEDDER_MODEL = os.environ.get('EMBEDDER_MODEL', "nomic-embed-text")
EMBEDDING_DIM = os.environ.get('EMBEDDING_DIM', 384)

print(NEO4J_URI)
print(NEO4J_USER)
print(NEO4J_PASSWORD)

print(LLM_API_KEY)
print(LLM_BASE_URL)
print(LLM_MODEL)

print(EMBEDDER_API_KEY)
print(EMBEDDER_BASE_URL)
print(EMBEDDER_MODEL)
print(EMBEDDING_DIM)