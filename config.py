"""
Configuration for YouTube RAG system.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Endee Settings
INDEX_NAME = "yt_transcripts"
EMBED_DIM = 1536  # text-embedding-3-small dimension
ENDEE_URL = "http://localhost:8080"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions
LLM_MODEL = "openai/gpt-4o-mini"

# Chunking
CHUNK_WINDOW_SEC = 30
CHUNK_OVERLAP_SEC = 5

# Retrieval
TOP_K = 5