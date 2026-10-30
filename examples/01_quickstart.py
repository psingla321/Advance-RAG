"""Quickstart: index some text and ask questions. No API key, no containers.

    python examples/01_quickstart.py
"""

from app.rag import RAG

POLICIES = [
    "The nightly hotel limit is 200 USD in standard cities and 350 USD in high-cost cities "
    "such as London, New York, Tokyo and Zurich. Stays above the limit require written "
    "approval from a vice president before booking.",
    "Hybrid employees must work from the office at least two days per week. Fully remote "
    "arrangements require VP sign-off and come with a 500 USD home office stipend.",
    "Passwords must be rotated every 90 days. Multi-factor authentication is mandatory for "
    "all administrative accounts and for any access from outside the corporate network.",
    "The Software Engineer II salary band is 120,000 to 160,000 USD annually. Senior "
    "Engineer is 160,000 to 210,000 USD. Bands are reviewed each January.",
]


def main() -> None:
    # "offline" uses a hashing embedder and an extractive generator: no network at all.
    # Swap to RAG("hybrid") once you have an LLM configured.
    rag = RAG("offline")
    print(rag, "\n")

    report = rag.add_texts(POLICIES)
    print(f"Indexed {report['chunks']} chunks from {report['documents']} documents.\n")

    for question in (
        "What is the hotel limit in high-cost cities?",
        "How often do I need to change my password?",
        "What is the salary band for a Senior Engineer?",
        "What is the company holiday in March?",  # not in the corpus: should abstain
    ):
        rag.print_answer(question)
        print("-" * 70)

    # Everything the pipeline did is inspectable.
    explanation = rag.explain("What is the hotel limit in high-cost cities?")
    print("\nPer-stage timings:")
    for stage in explanation["stages"]:
        print(f"  {stage['stage']:<10} {stage['ms']:>7.2f} ms")

    rag.close()


if __name__ == "__main__":
    main()
