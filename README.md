# ENDEE-YT: YouTube Video Q&A with RAG

A Retrieval-Augmented Generation (RAG) system for querying YouTube video transcripts using Endee vector database for semantic search and OpenRouter LLM for answer generation.

## Problem Statement

YouTube videos contain valuable information, but finding specific content within long videos is time-consuming. This project enables semantic search across video transcripts, allowing users to ask questions and receive accurate answers with timestamp references to the exact moments in the video.

## System Architecture

### Overview
```
YouTube URL → Transcript Extraction → Chunking → Embedding → Endee Vector Store
                                                                        ↓
User Question → Embedding → Similarity Search (Endee) → Context Retrieval → LLM → Answer + Timestamps
```

### Components

**1. Ingestion Pipeline (`ingest.py`)**
- Extracts YouTube video transcripts via `youtube-transcript-api`
- Splits transcripts into time-aware chunks with overlap
- Generates embeddings using OpenAI's `text-embedding-3-small` (1536 dimensions)
- Stores vectors in Endee with metadata (timestamps, text, video ID)

**2. Query Pipeline (`query.py`)**
- Embeds user questions using the same model
- Retrieves top-k similar chunks from Endee
- Constructs context-aware prompts with explicit timestamp markers
- Generates answers via OpenRouter LLM with strict source attribution

**3. Vector Database (`vector_db.py`)**
- Endee as the core vector store
- HNSW algorithm for fast approximate nearest neighbor search
- Cosine similarity metric for semantic matching
- INT8D precision for speed-accuracy balance

## How Endee is Used

### Index Configuration
```python
client.create_index(
    name="yt_transcripts",
    dimension=1536,          # text-embedding-3-small dimension
    space_type="cosine",     # Cosine similarity for semantic search
    precision=Precision.INT8D # Quantization for faster queries
)
```

### Vector Storage
Each chunk is stored with:
- **ID**: `{video_id}_{chunk_index}` for unique identification
- **Vector**: 1536-dimensional embedding from OpenAI
- **Metadata**: 
  - `video_id`: Source video identifier
  - `chunk_index`: Sequential position in transcript
  - `start_time` / `end_time`: Exact timestamp boundaries
  - `text`: Original transcript content

### Similarity Search
```python
results = index.query(
    vector=query_embedding,
    top_k=5  # Retrieve 5 most relevant chunks
)
```

Endee's cosine similarity metric ensures semantically related content ranks highest, even when exact keywords don't match.

### Key Advantages
- **Fast ANN Search**: HNSW algorithm provides sub-linear query time
- **Metadata Filtering**: Can extend to filter by video_id, date, or custom tags
- **Precision Trade-off**: INT8D quantization reduces storage by 4x vs FLOAT32 while maintaining accuracy
- **Scalability**: Handles millions of vectors across unlimited videos

## Technical Implementation

### Chunking Strategy (`chunking.py`)
- **Window Size**: 30 seconds of transcript per chunk
- **Overlap**: 5 seconds between consecutive chunks
- Preserves context across chunk boundaries
- Maintains precise timestamp mapping

### Embedding Pipeline (`embeddings.py`)
- Batch processing (10 texts per API call) for efficiency
- Automatic retry logic with exponential backoff
- Uses `text-embedding-3-small` (cost-effective, high quality)

### Deduplication
Videos are checked before ingestion using Endee's `get_vector()` method to prevent redundant processing.

## Setup and Execution

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- OpenRouter API key (for embeddings and LLM. Store it in .env!)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ArjunJagdale/ENDEE-YT.git
cd ENDEE-YT
```

2. **Start Endee server**
```bash
docker compose up -d
```

Verify Endee is running at `http://localhost:8080`

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
Create `.env` file:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Usage

**Launch the UI**
```bash
python ui.py
```

Access Gradio interface at `http://localhost:7860`

**Ingest Tab**
1. Paste YouTube URL (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
2. Click "Ingest"
3. System processes transcript, creates chunks, generates embeddings, and stores in Endee

**Ask Tab**
1. Enter question (e.g., "What does the speaker say about machine learning?")
2. Click "Ask"
3. Receive answer with source timestamps from relevant video segments

**Reset Database** (if needed)
```bash
python reset_index.py
```

### Configuration (`config.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `INDEX_NAME` | `yt_transcripts` | Endee index identifier |
| `EMBED_DIM` | `1536` | text-embedding-3-small dimension |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Answer generation model |
| `CHUNK_WINDOW_SEC` | `30` | Chunk duration in seconds |
| `CHUNK_OVERLAP_SEC` | `5` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |

## Technical Highlights

### Why Endee?
- **Performance**: HNSW indexing provides millisecond-level query times even with 100k+ vectors
- **Simplicity**: Clean Python SDK with minimal boilerplate
- **Efficiency**: INT8D quantization reduces memory footprint without sacrificing accuracy
- **Metadata Support**: Rich filtering capabilities for future enhancements (e.g., search within specific videos or date ranges)
- **Local-First**: No vendor lock-in, runs entirely on-premise via Docker

### Workflow Efficiency
1. **Batch Embeddings**: 10x faster than sequential API calls
2. **Existence Check**: Prevents redundant processing of already-ingested videos
3. **Timestamp Preservation**: Maintains exact source attribution for answers
4. **Error Handling**: Automatic retries for transient API failures

## Future Enhancements
- Multi-video search across entire channels
- Temporal filtering (search within date ranges)
- Hybrid search combining dense vectors (semantic) and sparse vectors (keyword)
- Video summarization using retrieved chunks
- Support for multiple languages via multilingual embeddings

## Project Structure
```
ENDEE-YT/
├── ingest.py          # Video ingestion pipeline
├── query.py           # Question answering logic
├── chunking.py        # Transcript chunking utilities
├── embeddings.py      # Embedding generation
├── vector_db.py       # Endee client wrapper
├── ui.py              # Gradio web interface
├── config.py          # Configuration parameters
├── reset_index.py     # Database reset utility
├── docker-compose.yml # Endee server configuration
└── requirements.txt   # Python dependencies
```

## Dependencies
- `endee`: Vector database client
- `youtube-transcript-api`: Transcript extraction
- `gradio`: Web UI framework
- `requests`: HTTP client for OpenRouter API
- `python-dotenv`: Environment management

## Author
**Arjun Jagdale**  
GitHub: [@ArjunJagdale](https://github.com/ArjunJagdale)

## License
MIT
