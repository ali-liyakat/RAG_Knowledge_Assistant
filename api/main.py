"""
main.py — FastAPI backend exposing the RAG pipeline.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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