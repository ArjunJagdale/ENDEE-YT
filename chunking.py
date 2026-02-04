"""
Timestamp-aware transcript chunking.
"""

from typing import List, Dict
from config import CHUNK_WINDOW_SEC, CHUNK_OVERLAP_SEC


def chunk_transcript(segments: List[Dict]) -> List[Dict]:
    """
    Convert transcript segments into overlapping time-aware chunks.
    
    Args:
        segments: List of dicts with 'text', 'start', 'duration'
    
    Returns:
        List of chunks with 'text', 'start_time', 'end_time'
    """
    if not segments:
        return []
    
    chunks = []
    current_text = []
    chunk_start = None
    chunk_end = None
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        start = float(seg.get("start", 0))
        end = start + float(seg.get("duration", 0))
        
        # Initialize first chunk
        if chunk_start is None:
            chunk_start = start
        
        current_text.append(text)
        chunk_end = end
        
        # Check if we've reached the window size
        if chunk_end - chunk_start >= CHUNK_WINDOW_SEC:
            # Save current chunk
            chunks.append({
                "text": " ".join(current_text),
                "start_time": chunk_start,
                "end_time": chunk_end,
            })
            
            # Start new chunk with overlap
            overlap_time = chunk_end - CHUNK_OVERLAP_SEC
            
            # Find segments that fall in overlap period
            overlap_texts = []
            overlap_start = None
            
            for s in segments:
                s_start = float(s.get("start", 0))
                s_end = s_start + float(s.get("duration", 0))
                
                if s_end > overlap_time:
                    s_text = s.get("text", "").strip()
                    if s_text:
                        overlap_texts.append(s_text)
                        if overlap_start is None:
                            overlap_start = s_start
            
            current_text = overlap_texts
            chunk_start = overlap_start if overlap_start else chunk_end
    
    # Add final chunk
    if current_text and chunk_start is not None:
        chunks.append({
            "text": " ".join(current_text),
            "start_time": chunk_start,
            "end_time": chunk_end,
        })
    
    return chunks
