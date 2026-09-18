"""
retriever.py — embeds a query and retrieves the most relevant chunks from Qdrant.
"""

from src.ingestion.embedder import get_embedding_model, get_qdrant_client, COLLECTION_NAME


def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.5) -> list[dict]:
    """
    Given a user question, returns the top_k most relevant chunks from Qdrant.

    Returns a list of dicts: {"source", "page", "text", "score"}
    """
    model = get_embedding_model()
    client = get_qdrant_client()

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
    ).points

    retrieved = [
        {
            "source": point.payload["source"],
            "page": point.payload["page"],
            "text": point.payload["text"],
            "score": point.score,
        }
        for point in results
    ]

    return retrieved


if __name__ == "__main__":
    test_query = "What is precision and recall?"
    results = retrieve(test_query)

    print(f"Query: {test_query}")
    print(f"Retrieved {len(results)} chunk(s):\n")

    for i, r in enumerate(results, start=1):
        print(f"--- Result {i} (score: {r['score']:.3f}) ---")
        print(f"Source: {r['source']}, Page: {r['page']}")
        print(r['text'][:200], "...\n")