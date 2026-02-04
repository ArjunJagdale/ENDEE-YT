"""
Question answering using RAG.
"""

from typing import Dict
import requests
from embeddings import embed_text
from vector_db import get_db
from config import LLM_MODEL, OPENROUTER_API_KEY, TOP_K


SYSTEM_PROMPT = """Answer questions using ONLY the provided context.
DO NOT mention timestamps in your answer - they will be shown separately.

CRITICAL RULES:
1. Answer ONLY using the provided transcript context
2. Be concise and accurate"""


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def answer_question(question: str) -> Dict:
    """
    Answer a question using RAG.
    
    Steps:
    1. Embed the question
    2. Retrieve relevant chunks
    3. Format context with clear timestamps
    4. Query LLM with strict instructions
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")
    
    # Embed question
    query_vector = embed_text(question)
    
    # Retrieve relevant chunks
    db = get_db()
    results = db.query(query_vector, top_k=TOP_K)
    print(f"✓ Retrieved {len(results)} relevant chunks")
    
    # Format context with very clear timestamp markers
    context_parts = []
    for i, result in enumerate(results, 1):
        meta = result.get("meta", {})
        start = format_timestamp(meta.get("start_time", 0))
        end = format_timestamp(meta.get("end_time", 0))
        text = meta.get("text", "")
        
        context_parts.append(
            f"--- Excerpt {i} ---\n"
            f"Timestamp: {start} to {end}\n"
            f"Content: {text}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Build user prompt with explicit timestamp instruction
    user_prompt = f"""Context from YouTube transcript:

{context}

Question: {question}

IMPORTANT: 
- Use ONLY the timestamps shown above (e.g., "{format_timestamp(results[0].get('meta', {}).get('start_time', 0))}")
- Do NOT make up or approximate timestamps
- If you reference content, cite its exact timestamp from above"""
    
    # Query LLM
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Lower temperature for more factual responses
        },
        timeout=60,
    )
    
    if response.status_code != 200:
        raise Exception(f"LLM request failed: {response.text}")
    
    answer = response.json()["choices"][0]["message"]["content"]
    
    # Extract timestamps from retrieved chunks
    timestamps = [
        {
            "start": format_timestamp(r.get("meta", {}).get("start_time", 0)),
            "end": format_timestamp(r.get("meta", {}).get("end_time", 0)),
            "similarity": r.get("similarity", 0),
        }
        for r in results
    ]
    
    return {
        "answer": answer,
        "timestamps": timestamps,
        "sources": len(results),
    }