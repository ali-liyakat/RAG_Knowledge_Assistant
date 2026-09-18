"""
llm.py — wraps the Groq API for grounded answer generation.
"""

import os
from dotenv import load_dotenv
from groq import Groq

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

_client = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Sends the question + retrieved context to Groq and returns a grounded answer.
    """
    client = get_groq_client()
    user_prompt = build_user_prompt(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from src.retrieval.retriever import retrieve

    test_question = "What is precision and recall?"
    chunks = retrieve(test_question)
    answer = generate_answer(test_question, chunks)

    print(f"Question: {test_question}\n")
    print("Answer:")
    print(answer)