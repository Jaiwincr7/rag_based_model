import os
from langchain_community.llms import HuggingFaceEndpoint, HuggingFacePipeline


model_name = os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
use_remote_llm = os.getenv("USE_REMOTE_LLM", "true").lower() == "true"
_llm = None

def wrap_chat(prompt: str) -> str:
    return f"""<|system|>
You are a cybersecurity expert.
<|user|>
{prompt}
<|assistant|>
"""

def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    if use_remote_llm:
        # Remote inference keeps memory usage low on small Render instances.
        _llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            max_new_tokens=120,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            task="text-generation",
        )
        return _llm

    # Optional local fallback for larger instances.
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
            f"{d.page_content.strip()}"
        )
    return "\n\n".join(formatted)

