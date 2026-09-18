"""
prompts.py — prompt templates for grounded, citation-aware answers.
"""

SYSTEM_PROMPT = """You are a study assistant that answers questions using ONLY the provided context from the user's own notes.

Rules:
- Answer strictly using the given context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly — do not guess or make things up.
- After your answer, list the sources you used in this exact format: [source_filename, page X].
- Keep answers clear and concise, matching the technical level of the context.
"""


def build_user_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Builds the user-turn prompt: the question + formatted retrieved context.
    """
    if not retrieved_chunks:
        return f"""Question: {question}

Context: (no relevant context was found in the notes)

Answer that you don't have enough information from the notes to answer this."""

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Chunk {i} — Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    return f"""Context from notes:
{context_text}

Question: {question}

Answer the question using only the context above, then list your sources."""