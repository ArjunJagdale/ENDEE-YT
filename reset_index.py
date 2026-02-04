"""
Reset the Endee index.
Use this if you need to change dimensions or start fresh.
"""

from vector_db import get_db
from config import INDEX_NAME

def reset_index():
    """Delete and recreate the index."""
    try:
        db = get_db()
        db.delete_index()
        print(f"✓ Deleted index: {INDEX_NAME}")
        print("✓ Index will be recreated on next ingestion")
    except Exception as e:
        print(f"Note: {e}")
        print("Index may not exist yet - this is fine!")

if __name__ == "__main__":
    print("⚠️  This will delete all ingested videos from the database!")
    confirm = input("Type 'yes' to continue: ")
    
    if confirm.lower() == 'yes':
        reset_index()
    else:
        print("Cancelled.")