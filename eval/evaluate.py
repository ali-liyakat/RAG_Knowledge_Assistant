"""
evaluate.py — measures retrieval quality against a known-answer question set.
"""

import json
from pathlib import Path
from src.retrieval.retriever import retrieve

EVAL_FILE = Path(__file__).parent / "eval_questions.json"


def run_evaluation(top_k: int = 8):
    with open(EVAL_FILE) as f:
        eval_set = json.load(f)

    hits = 0
    results = []

    for item in eval_set:
        question = item["question"]
        expected_source = item["expected_source"]
        expected_page = item["expected_page"]

        retrieved = retrieve(question, top_k=top_k)

        found = any(
            r["source"] == expected_source and r["page"] == expected_page
            for r in retrieved
        )

        if found:
            hits += 1

        results.append({
            "question": question,
            "expected": f"{expected_source}, page {expected_page}",
            "found": found,
            "retrieved": [f"{r['source']}, page {r['page']} (score {r['score']:.3f})" for r in retrieved],
        })

    accuracy = hits / len(eval_set) if eval_set else 0

    print(f"Retrieval accuracy: {hits}/{len(eval_set)} ({accuracy:.0%})\n")
    for r in results:
        status = "PASS" if r["found"] else "FAIL"
        print(f"[{status}] {r['question']}")
        print(f"  Expected: {r['expected']}")
        if not r["found"]:
            print(f"  Got instead:")
            for ret in r["retrieved"]:
                print(f"    - {ret}")
        print()

    return accuracy


if __name__ == "__main__":
    run_evaluation()