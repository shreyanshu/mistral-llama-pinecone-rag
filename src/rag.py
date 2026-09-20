from .retriever import retrieve
from .llm import generate


def answer(question: str):
    matches = retrieve(question)

    print("\nRetrieved context:")
    for i, match in enumerate(matches, start=1):
        print(
            f"  [{i}] {match['source']} "
            f"(score={match['score']:.4f})"
        )

    if matches:
        context = "\n\n".join(
            f"[{i}] Source: {m['source']}\n{m['text']}"
            for i, m in enumerate(matches, start=1)
        )
    else:
        context = "No relevant knowledge-base documents were retrieved."

    prompt = f"""Knowledge-base context:

{context}

Question:
{question}

Answer using only the knowledge-base context above.
If the answer is not present in the context, say so.
Do not invent information.
"""

    return generate(prompt)
