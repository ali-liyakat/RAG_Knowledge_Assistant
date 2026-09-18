"""
reset_collection.py — deletes and recreates the Qdrant collection from scratch.
Use when duplicate/orphaned points need to be cleared out.
"""

from src.ingestion.embedder import get_qdrant_client, ensure_collection, COLLECTION_NAME

client = get_qdrant_client()

existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

ensure_collection(client)
print("Collection reset. Now re-run the bulk embedder to repopulate it.")