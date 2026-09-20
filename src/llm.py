import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from .config import HF_MODEL, MODEL, USE_4BIT, MAX_NEW_TOKENS

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model

    if _model is not None:
        return _tokenizer, _model

    print(f"Loading model: {MODEL}")
    print(f"Hugging Face checkpoint: {HF_MODEL}")

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        **kwargs,
    )

    return _tokenizer, _model


def generate(user_prompt: str) -> str:
    tokenizer, model = load_model()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a RAG assistant. "
                "Answer using only the supplied knowledge-base context. "
                "If the context does not contain the answer, say that "
                "the answer is not available in the knowledge base. "
                "Do not invent facts."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.01,
            top_p=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()