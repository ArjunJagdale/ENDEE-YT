"""
YouTube video ingestion pipeline.
"""

from typing import Dict
from youtube_transcript_api import YouTubeTranscriptApi
from chunking import chunk_transcript
from embeddings import embed_texts
from vector_db import get_db


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    raise ValueError("Invalid YouTube URL")


def check_video_exists(video_id: str) -> bool:
    """Check if video is already in the vector store."""
    try:
        db = get_db()
        # Try to get the first chunk of this video
        result = db.index.get_vector(f"{video_id}_0")
        return result is not None
    except:
        return False


def ingest_video(youtube_url: str) -> Dict:
    """
    Ingest a YouTube video into the vector database.
    
    Steps:
    1. Extract video ID
    2. Check if already exists
    3. If exists, return early
    4. Otherwise, fetch transcript, chunk, embed, and store
    """
    # Extract video ID
    video_id = extract_video_id(youtube_url)
    print(f"Processing video: {video_id}")
    
    # Check if already exists
    if check_video_exists(video_id):
        print(f"✓ Video {video_id} already in vector store")
        return {
            "video_id": video_id,
            "chunks": "existing",
            "url": youtube_url,
            "already_exists": True,
        }
    
    # Fetch transcript
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    segments = transcript.to_raw_data()
    print(f"✓ Fetched {len(segments)} transcript segments")
    
    # Chunk transcript
    chunks = chunk_transcript(segments)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Prepare vectors for database
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{video_id}_{i}",
            "vector": embedding,
            "meta": {
                "video_id": video_id,
                "chunk_index": i,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "text": chunk["text"],
            }
        })
    
    # Store in database
    db = get_db()
    db.upsert_vectors(vectors)
    print(f"✓ Stored {len(vectors)} vectors in database")
    
    return {
        "video_id": video_id,
        "chunks": len(chunks),
        "url": youtube_url,
        "already_exists": False,
    }