from dotenv import load_dotenv

load_dotenv()

from .config import (
    MODEL,
    HF_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_CONFIG,
    PINECONE_INDEX,
)

from .rag import answer

def main():

    print("=" * 60)
    print(" Configurable Local LLM + Pinecone RAG")
    print("=" * 60)

    print(f"LLM: {MODEL}")
    print(f"Checkpoint:{HF_MODEL}")

    print(f"Embedding: {EMBEDDING_MODEL}")

    print(f"Embedding model: {EMBEDDING_CONFIG['model']}")

    print(f"Pinecone:  {PINECONE_INDEX}")

    print("Type 'exit' or 'quit' to stop.")

    print()

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit",}:
            break

        if not question:
            continue

        response = answer(question)

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()