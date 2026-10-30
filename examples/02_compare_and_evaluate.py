"""Measure instead of guessing: evaluate one pipeline, then compare several.

    python examples/02_compare_and_evaluate.py

Shows the loop this toolkit exists for — change one component, see the number move.
"""

from pathlib import Path

from app.rag import RAG, Dataset, Sweep

CORPUS = Path(__file__).resolve().parents[1] / "backend" / "sample_data"

GOLDEN = Dataset.from_pairs(
    [
        ("What is the nightly hotel limit in high-cost cities?", "300 USD"),
        ("How many days per week must hybrid employees be in the office?", "at least two days"),
        ("How often must passwords be rotated?", "every 90 days"),
        ("What equipment does the company provide for remote work?", "a laptop, a monitor and a headset"),
        ("Who must approve stays above the hotel limit?", "a vice president"),
    ],
    name="policies",
)


def main() -> None:
    # --- 1. Evaluate retrieval on its own -------------------------------
    # Always start here: if the right passage is not retrieved, no prompt can save you.
    rag = RAG("hybrid")
    rag.add_directory(CORPUS, show_progress=False)
    print(f"Indexed {rag.count()} chunks.\n")

    print("Retrieval quality at several cut-offs:")
    for key, value in rag.evaluate_retrieval(GOLDEN, k_values=[1, 3, 5, 10]).items():
        print(f"  {key:<16} {value}")

    # --- 2. Evaluate the whole pipeline ---------------------------------
    print("\nEnd-to-end evaluation:")
    report = rag.evaluate(GOLDEN, show_progress=False)
    print(report.table())

    print("\nThe two weakest samples — read these, they tell you what to fix:")
    for sample in report.worst(2):
        print(f"\n  Q: {sample.question}")
        print(f"  got:      {sample.answer[:120]}")
        print(f"  expected: {sample.ground_truth}")
    rag.close()

    # --- 3. Compare strategies ------------------------------------------
    # Each configuration re-indexes the corpus and is scored on the same questions.
    print("\n\nComparing retrieval strategies on your own data:\n")
    sweep = Sweep("hybrid", GOLDEN, corpus=CORPUS, metrics=["hit_rate", "mrr", "ndcg", "context_precision"])
    sweep.grid(**{"retriever.type": ["dense", "bm25", "hybrid_rrf", "mmr"]})
    results = sweep.run(save=False, show_progress=False)
    print(results.leaderboard())

    # --- 4. Ablate one knob at a time -----------------------------------
    # Linear instead of exponential: tells you which knob matters, not the global optimum.
    print("\n\nAblation — which single change helps most?\n")
    ablation = Sweep("hybrid", GOLDEN, corpus=CORPUS, metrics=["hit_rate", "ndcg", "context_precision"])
    ablation.ablate(**{"top_n": [3, 5, 12], "chunker.max_tokens": [200, 500]})
    print(ablation.run(save=False, show_progress=False).leaderboard())


if __name__ == "__main__":
    main()
