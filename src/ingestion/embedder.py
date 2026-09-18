"""
embedder.py — embeds text chunks and stores them in Qdrant.
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

COLLECTION_NAME = "ml_dl_notes"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384  # bge-small-en-v1.5 output dimension

_model = None  # loaded once, reused


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,  # generous timeout for slower connections
    )


def ensure_collection(client: QdrantClient):
    """Create the collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"Creating Qdrant collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists — will add to it.")


def embed_and_store(chunks: list[dict]):
    """
    Embeds a list of chunk dicts and upserts them into Qdrant.
    Each chunk dict needs: source, page, chunk_id, text.
    """
    model = get_embedding_model()
    client = get_qdrant_client()
    ensure_collection(client)

    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    points = [
        PointStruct(
            id=chunk["chunk_id"],
            vector=vector.tolist(),
            payload={
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    batch_size = 50
    print(f"Uploading to Qdrant in batches of {batch_size} ...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        print(f"  Uploaded {min(i + batch_size, len(points))}/{len(points)}")

    print(f"Done. {len(points)} chunks stored in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    from src.ingestion.loader import load_all_pdfs
    from src.ingestion.chunker import chunk_pages

    pages = load_all_pdfs()
    chunks = chunk_pages(pages)
    embed_and_store(chunks)