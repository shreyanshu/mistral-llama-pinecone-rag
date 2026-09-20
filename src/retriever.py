from pinecone import Pinecone, ServerlessSpec


from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX,
    NAMESPACE,
    TOP_K,
    EMBEDDING_CONFIG,
)

from .embeddings import encode


def retrieve(
    question: str,
    top_k: int = TOP_K,
):

    if not PINECONE_API_KEY:

        raise RuntimeError(
            "PINECONE_API_KEY is missing."
        )

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )


    existing = [x["name"] if isinstance(x, dict) else x.name for x in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_CONFIG["dimension"],
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(
        PINECONE_INDEX
    )

    # ========================================================
    # Pinecone integrated embedding
    # ========================================================

    if (EMBEDDING_CONFIG["type"]== "pinecone_integrated"):

        result = index.search(
            namespace=NAMESPACE,
            query={
                "inputs": {
                    "text": question
                },
                "top_k": top_k,
            },
            fields=[
                "text",
                "source",
            ],
        )

        matches = []

        for hit in result.result.hits:

            fields = hit.fields or {}

            matches.append({

                "id": hit["_id"],

                "score": hit["_score"],

                "text": fields.get(
                    "text",
                    ""
                ),

                "source": fields.get(
                    "source",
                    "unknown"
                ),

            })

        return matches

    # ========================================================
    # Local embedding model
    # ========================================================

    query_vector = encode(
        [question]
    )[0]

    result = index.query(

        namespace=NAMESPACE,

        vector=query_vector,

        top_k=top_k,

        include_metadata=True,

    )

    matches = []

    for match in result["matches"]:
        metadata = match.get(
            "metadata",
            {}
        )
        matches.append({
            "id": match["id"],
            "score": match["score"],

            "text": metadata.get(
                "text",
                ""
            ),

            "source": metadata.get(
                "source",
                "unknown"
            ),

        })

    return matches