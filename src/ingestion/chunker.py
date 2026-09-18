"""
chunker.py — splits extracted page text into overlapping chunks for embedding.
"""

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

NAMESPACE = uuid.NAMESPACE_DNS


def chunk_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Splits each page's text into overlapping chunks.
    Keeps source/page metadata attached to every chunk.

    chunk_id is a deterministic UUID derived from (source, page, position) —
    stable across re-runs, so re-uploading the same PDF updates its own
    chunks instead of colliding with or overwriting other files' chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for local_idx, chunk_text in enumerate(page_chunks):
            unique_key = f"{page['source']}_p{page['page']}_c{local_idx}"
            chunk_id = str(uuid.uuid5(NAMESPACE, unique_key))

            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })

    return chunks


if __name__ == "__main__":
    from src.ingestion.loader import load_all_pdfs

    pages = load_all_pdfs()
    chunks = chunk_pages(pages)

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print("\n--- Sample chunk ---")
        print(f"Source: {chunks[0]['source']}, Page: {chunks[0]['page']}, Chunk ID: {chunks[0]['chunk_id']}")
        print(chunks[0]['text'])
        print(f"\nChunk length: {len(chunks[0]['text'])} chars")