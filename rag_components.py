from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch

os.environ["HF_TOKEN"] = "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"

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

def wrap_chat(prompt: str) -> str:
    return f"""<|system|>
You are a cybersecurity expert.
<|user|>
{prompt}
<|assistant|>
"""

gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.3,
    top_p=0.9,                     
    do_sample=True,               
    repetition_penalty=1.2,        
    no_repeat_ngram_size=4,        
    return_full_text=False
)

from langchain_community.llms import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=gen_pipeline)

def format_docs(docs):
    formatted = []
    for d in docs:
        formatted.append(
            f"[SOURCE: {d.metadata.get('source')}\n"
            f"{d.page_content.strip()}"
        )
    return "\n\n".join(formatted)

