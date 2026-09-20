from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_CONFIG


_model = None


def get_local_model():

    global _model

    if _model is None:

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_CONFIG['model']}"
        )

        _model = SentenceTransformer(
            EMBEDDING_CONFIG["model"]
        )

    return _model


def encode(texts):

    """
    Generate embeddings using a local
    Sentence Transformers model.

    This function is only used when:

        EMBEDDING_MODEL = "minilm"
    """

    if EMBEDDING_CONFIG["type"] != "local":

        raise RuntimeError(
            "Local encode() called while using "
            "a Pinecone integrated embedding model."
        )

    model = get_local_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    return embeddings.tolist()