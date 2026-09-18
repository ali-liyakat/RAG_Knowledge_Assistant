"""
chunker.py — splits extracted page text into overlapping chunks for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Splits each page's text into overlapping chunks.
    Keeps source/page metadata attached to every chunk.

    Returns a list of dicts: {"source": ..., "page": ..., "chunk_id": ..., "text": ...}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_counter,
                "text": chunk_text
            })
            chunk_counter += 1

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