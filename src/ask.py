import sys
from dotenv import load_dotenv

load_dotenv()

from .config import MODEL, HF_MODEL
from .rag import answer


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python -m src.ask "your question"'
        )

    question = " ".join(sys.argv[1:])

    print(f"Model: {MODEL}")
    print(f"Checkpoint: {HF_MODEL}")
    print(f"Question: {question}")

    print("\nAnswer:")
    print(answer(question))


if __name__ == "__main__":
    main()
