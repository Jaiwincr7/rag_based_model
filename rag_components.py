import os
from langchain_community.llms import HuggingFaceEndpoint, HuggingFacePipeline

model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
use_remote_llm = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"
_llm = None


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if use_remote_llm:
        _llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "80")),
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            task="text-generation",
            timeout=int(os.getenv("HF_TIMEOUT_SEC", "60")),
        )
        return _llm

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        return_full_text=False,
    )
    _llm = HuggingFacePipeline(pipeline=gen_pipeline)
    return _llm


def format_docs(docs):
    formatted = []
    for d in docs:
        formatted.append(
            f"[SOURCE: {d.metadata.get('source')}\n"
            f"{d.page_content.strip()[:500]}"
        )
    return "\n\n".join(formatted)
