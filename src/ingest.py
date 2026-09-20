from pathlib import Path

from pinecone import Pinecone

from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    PINECONE_CLOUD,
    PINECONE_REGION,
    NAMESPACE,
    EMBEDDING_CONFIG,
)

from .embeddings import encode


KB_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "kb"
)


def create_index(pc):

    if PINECONE_INDEX in pc.list_indexes().names():

        print(
            f"Using existing index: "
            f"{PINECONE_INDEX}"
        )

        return

    print(
        f"Creating index: {PINECONE_INDEX}"
    )

    # --------------------------------------------------------
    # Pinecone integrated embedding
    # --------------------------------------------------------

    if EMBEDDING_CONFIG["type"] == "pinecone_integrated":

        pc.indexes.create_for_model(
            name=PINECONE_INDEX,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
            embed={
                "model": EMBEDDING_CONFIG["model"],
                "field_map": {
                    "text": "text"
                },
                "metric": "cosine",
            },
        )

    # --------------------------------------------------------
    # Local embedding model
    # --------------------------------------------------------

    else:

        pc.create_index(

            name=PINECONE_INDEX,

            dimension=EMBEDDING_CONFIG["dimension"],

            metric="cosine",

            spec={
                "serverless": {
                    "cloud": PINECONE_CLOUD,
                    "region": PINECONE_REGION,
                }
            },
        )


def read_documents():

    documents = []

    for path in sorted(
        KB_DIR.glob("*.txt")
    ):

        documents.append({

            "id": path.stem,

            "text": path.read_text(
                encoding="utf-8"
            ).strip(),

            "source": path.name,

        })

    if not documents:

        raise RuntimeError(
            f"No .txt files found in {KB_DIR}"
        )

    return documents


def ingest_integrated(
    pc,
    documents,
):

    index = pc.Index(
        PINECONE_INDEX
    )

    records = [

        {
            "_id": doc["id"],
            "text": doc["text"],
            "source": doc["source"],
        }

        for doc in documents
    ]

    index.upsert_records(

        namespace=NAMESPACE,

        records=records,

    )


def ingest_local(
    pc,
    documents,
):

    index = pc.Index(
        PINECONE_INDEX
    )

    texts = [
        doc["text"]
        for doc in documents
    ]

    vectors = encode(texts)

    records = []

    for doc, vector in zip(
        documents,
        vectors,
    ):

        records.append({

            "id": doc["id"],

            "values": vector,

            "metadata": {

                "text": doc["text"],

                "source": doc["source"],

            },

        })

    index.upsert(

        namespace=NAMESPACE,

        vectors=records,

    )


def main():

    if not PINECONE_API_KEY:

        raise RuntimeError(
            "PINECONE_API_KEY is missing. "
            "Put it in your .env file."
        )

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    create_index(pc)

    documents = read_documents()

    if (
        EMBEDDING_CONFIG["type"]
        == "pinecone_integrated"
    ):

        ingest_integrated(
            pc,
            documents,
        )

    else:

        ingest_local(
            pc,
            documents,
        )

    print()
    print("Ingestion complete.")
    print(
        f"Index: {PINECONE_INDEX}"
    )
    print(
        f"Embedding: "
        f"{EMBEDDING_CONFIG['model']}"
    )
    print(
        f"Documents: "
        f"{len(documents)}"
    )


if __name__ == "__main__":
    main()