"""
Vector database operations using Endee.
"""

from typing import List, Dict
from endee import Endee, Precision
from config import INDEX_NAME, EMBED_DIM, ENDEE_URL


class VectorDB:
    """Simplified vector database interface."""
    
    def __init__(self):
        self.client = Endee()
        self._ensure_index()
    
    def _ensure_index(self):
        """Create index if it doesn't exist, otherwise get existing."""
        try:
            # Try to get existing index
            self.index = self.client.get_index(INDEX_NAME)
            print(f"✓ Using existing index: {INDEX_NAME}")
        except:
            # Index doesn't exist, create it
            self.client.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                space_type="cosine",
                precision=Precision.INT8D
            )
            self.index = self.client.get_index(INDEX_NAME)
            print(f"✓ Created new index: {INDEX_NAME}")
    
    def upsert_vectors(self, vectors: List[Dict]):
        """Insert or update vectors in the index."""
        self.index.upsert(vectors)
    
    def query(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        """Query for similar vectors."""
        return self.index.query(vector=vector, top_k=top_k)
    
    def delete_index(self):
        """Delete the entire index."""
        self.client.delete_index(INDEX_NAME)
        print(f"✓ Deleted index: {INDEX_NAME}")


# Singleton instance
_db = None

def get_db() -> VectorDB:
    """Get or create the vector database instance."""
    global _db
    if _db is None:
        _db = VectorDB()
    return _db
