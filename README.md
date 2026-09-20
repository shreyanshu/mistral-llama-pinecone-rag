# Mistral 7B / Llama 3 + Pinecone RAG

A configurable local RAG project.

The same Pinecone knowledge base and RAG pipeline can run with either:

- Mistral 7B Instruct v0.3
- Meta Llama 3 8B Instruct

## Architecture

Question
  -> Pinecone semantic retrieval
  -> retrieved KB context
  -> selected Hugging Face model
  -> answer

Only the generation model changes. Pinecone and the RAG pipeline remain the same.

## 1. Environment

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configure Pinecone

Copy `.env.example` to `.env` and add:

```text
PINECONE_API_KEY=your_key
```

The project uses a Pinecone integrated-embedding index, so the demo does not need a separate embedding model.

## 3. Select the model

Open:

`src/config.py`

For Mistral:

```python
MODEL = "mistral"
```

For Llama 3:

```python
MODEL = "llama3"
```

The mapping is:

```python
MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}
```

## 4. Populate the sample KB

```powershell
python -m src.ingest
```

The sample documents are in:

```text
data/kb/
```

## 5. Run a question

```powershell
python -m src.ask "What is the refund window?"
```

## 6. Run interactive chat

```powershell
python -m src.chat
```

## 7. Switch model

Stop the program, change `MODEL` in `src/config.py`, then run it again.

Mistral:

```python
MODEL = "mistral"
```

Llama 3:

```python
MODEL = "llama3"
```

The RAG/Pinecone code does not change.

## Important: Llama 3 access

Meta Llama 3 model repositories on Hugging Face may require accepting the model's license/access conditions and authenticating with Hugging Face.

If required:

```powershell
huggingface-cli login
```

Then authenticate using your Hugging Face token.

## 4-bit inference

The default is:

```python
USE_4BIT = True
```

This uses bitsandbytes NF4 quantization and is intended to make local 7B/8B inference more practical on GPUs with limited VRAM.

## Why apply_chat_template()?

Mistral and Llama use different chat/instruction formatting.

The project therefore does not hard-code either model's special tokens.

Instead:

```python
tokenizer.apply_chat_template(...)
```

lets the selected Hugging Face tokenizer construct the correct prompt format.

## Project structure

```text
mistral-llama-pinecone-rag/
├── data/
│   └── kb/
│       ├── company_overview.txt
│       ├── refund_policy.txt
│       └── support_policy.txt
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── llm.py
│   ├── rag.py
│   ├── ask.py
│   └── chat.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Next step

A natural next version is:

PDF -> parser -> chunker -> Pinecone -> Mistral/Llama

with metadata such as:

- document
- page
- chunk_id
- section
- source
