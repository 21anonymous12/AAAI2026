from transformers import AutoTokenizer
from langchain_ollama import ChatOllama

def measure_and_adjust_context(prompt, model_name="microsoft/phi-4", max_context=16000):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)
    
    adjusted_context = min(token_count, max_context) + 2048
    
    return {
        "token_count": token_count,
        "adjusted_context": adjusted_context,
        "truncated": adjusted_context > max_context
    }


def llm_adjusted_context(llm, ctx_window):
    llm = ChatOllama(
    model="phi4:14b", 
    format="json",  
    temperature=0,
    num_ctx = ctx_window,  
    )
    return llm