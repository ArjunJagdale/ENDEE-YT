"""
Embedding utilities using OpenRouter API with batching for speed.
"""

import requests
import time
from typing import List
from config import EMBEDDING_MODEL, OPENROUTER_API_KEY


def embed_texts(texts: List[str], batch_size: int = 10, max_retries: int = 3) -> List[List[float]]:
    """
    Generate embeddings for texts with batching for faster processing.
    
    Instead of calling the API once per text, we batch multiple texts
    into a single API call, dramatically reducing total time.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    
    if not texts:
        return []
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # Process texts in batches
    for batch_num in range(0, len(texts), batch_size):
        batch = texts[batch_num:batch_num + batch_size]
        current_batch = (batch_num // batch_size) + 1
        
        print(f"  Processing batch {current_batch}/{total_batches} ({len(batch)} texts)...")
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": batch,  # Send multiple texts at once!
                    },
                    timeout=60,  # Longer timeout for batches
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Extract embeddings in order
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                elif response.status_code == 500:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  ⚠️  API error, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"API error after {max_retries} attempts")
                else:
                    raise Exception(f"API error {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print("Network error, retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"Network error: {str(e)}")
    
    return all_embeddings


def embed_text(text: str) -> List[float]:
    """Generate embedding for a single text."""
    return embed_texts([text])[0]