"""
pipeline.py — the single entry point that ties retrieval + generation together.
"""

from src.retrieval.retriever import retrieve
from src.generation.llm import generate_answer


def answer_question(question: str, top_k: int = 5) -> dict:
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "I couldn't find anything relevant to this question in your notes.",
            "sources": [],
        }

    answer = generate_answer(question, chunks)

    sources = [
        {"source": c["source"], "page": c["page"], "score": round(c["score"], 3)}
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    test_question = "What is the difference between accuracy and precision?"
    result = answer_question(test_question)

    print(f"Question: {test_question}\n")
    print("Answer:")
    print(result["answer"])
    print("\nSources used (with retrieval scores):")
    for s in result["sources"]:
        print(f"  {s['source']}, page {s['page']} (score: {s['score']})")