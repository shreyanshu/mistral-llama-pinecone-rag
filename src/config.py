import os

# ============================================================
# LLM CONFIGURATION
# ============================================================

MODEL = "qwen3"

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen3": "Qwen/Qwen3-4B",
    "qwen_small": "Qwen/Qwen2.5-0.5B-Instruct"
}

if MODEL not in MODELS:
    raise ValueError(
        f"Unsupported MODEL: {MODEL}. "
        f"Choose from: {list(MODELS.keys())}"
    )

HF_MODEL = MODELS[MODEL]


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

# Options:
#
# "pinecone_e5"
#     Pinecone-hosted multilingual-e5-large
#
# "minilm"
#     Local sentence-transformers/all-MiniLM-L6-v2
#
EMBEDDING_MODEL = "minilm"

EMBEDDINGS = {
    "pinecone_e5": {
        "type": "pinecone_integrated",
        "model": "multilingual-e5-large",
        "dimension": 1024,
    },

    "minilm": {
        "type": "local",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    },
}

if EMBEDDING_MODEL not in EMBEDDINGS:
    raise ValueError(
        f"Unsupported EMBEDDING_MODEL: {EMBEDDING_MODEL}. "
        f"Choose from: {list(EMBEDDINGS.keys())}"
    )

EMBEDDING_CONFIG = EMBEDDINGS[EMBEDDING_MODEL]


# ============================================================
# LOCAL LLM SETTINGS
# ============================================================

USE_4BIT = True
MAX_NEW_TOKENS = 300


# ============================================================
# PINECONE
# ============================================================

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

PINECONE_CLOUD = os.environ.get(
    "PINECONE_CLOUD",
    "aws"
)

PINECONE_REGION = os.environ.get(
    "PINECONE_REGION",
    "us-east-1"
)

# IMPORTANT:
# Different embedding models use different dimensions,
# so we use a separate index for each embedding configuration.

PINECONE_INDEX_PREFIX = "configurable-rag"

PINECONE_INDEX = (
    f"{PINECONE_INDEX_PREFIX}-{EMBEDDING_MODEL}"
)

NAMESPACE = "sample-kb"

TOP_K = 4