"""
main.py — FastAPI backend exposing the RAG pipeline.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import embed_and_store
from src.ingestion.embedder import get_qdrant_client, COLLECTION_NAME

from src.pipeline import answer_question

app = FastAPI(title="RAG Knowledge Assistant API")

# Allow the Streamlit frontend (running on a different port/domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a portfolio project; would restrict in a real production app
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class QuestionResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG Knowledge Assistant API is running"}


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    result = answer_question(request.question, top_k=request.top_k)
    return result


RAW_PDF_DIR = Path("data/raw_pdfs")
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    save_path = RAW_PDF_DIR / file.filename
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    pages = load_pdf(save_path)
    chunks = chunk_pages(pages)
    embed_and_store(chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "pages_processed": len(pages),
        "chunks_created": len(chunks),
    }


@app.get("/stats")
def get_stats():
    client = get_qdrant_client()
    total_chunks = client.count(collection_name=COLLECTION_NAME, exact=True).count

    sources = set()
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            offset=next_offset,
            with_payload=["source"],
        )
        for r in records:
            sources.add(r.payload["source"])
        if next_offset is None:
            break

    return {
        "total_chunks": total_chunks,
        "total_documents": len(sources),
        "documents": sorted(sources),
    }