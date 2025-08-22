import os
import json
import random
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import pipeline

import argparse

def parser():
    parser = argparse.ArgumentParser(description="Evaluate a TWE model.")
    parser.add_argument("--model", type=str)
    parser.add_argument("--adapter_path", type=str, default="./example_ckpt/tw_evaluator")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--max_count", type=int, default=1000)
    parser.add_argument("--start_idx", type=int, default=0)
    return parser.parse_args()

TWE_PROMPT = """
Below is a table header regarding {TITLE}:
Header: {HEADER}
You're tasked with answering the following question:
{QUESTION}
You have 2 answers with their corresponding reasoning processes derived by two different models. Answer [A] was derived by the A model. Answer [B] was derived by B model.
Reasoning of Answer [A] is: {RES1}
Answer [A] is: {ANS1}
Reasoning of Answer [B] is: {RES2}
Answer [B] is: {ANS2}
Your task is to determine whether this question can be correctly answered by these two models. True means yes and False means no.
Therefore, the final answer is:
"""

def main(args):
    adapter_path = args.adapter_path

    # Load the model and tokenizer
    compute_dtype = getattr(torch, "float16")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization_config=quant_config,
        device_map='auto'
    )

    model.load_adapter(adapter_path)

    evaluator = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=10,
        device_map="auto",
    )

    # Load the dataset
    with open(f"../outputs/{args.dataset}_{args.model}_result.jsonl", "r") as f:
        results = [json.loads(line) for line in f.readlines()]

    results = results[args.start_idx:args.start_idx + args.max_count]

    prompts = [
        TWE_PROMPT. format(
            TITLE=" ",
            HEADER=result["table"].splitlines()[0],
            QUESTION=result["question"],
            RES1=result["reason"],
            ANS1=result["pred"],
            RES2=result["num_solver_response"].split("```")[-1].strip(),
            ANS2=result["num_final_answer"]
        ) for result in results
    ]    

    # Create a dataset
    dataset = Dataset.from_dict({"text": prompts})
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Run the model on the dataset
    from tqdm import tqdm
    tqdm.write("Running the model...")
    predictions = []
    for batch in tqdm(dataloader):
        inputs = batch["text"]
        outputs = evaluator(inputs)
        for input, output in zip(inputs, outputs):
            predictions.append(output["generated_text"][len(input):].strip())

    # Save the predictions to a jsonl file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for idx, result in enumerate(results):
            result["tw_response"] = predictions[idx]
            result["tw"] = True if "true" in predictions[idx].lower().strip() else False
            f.write(json.dumps(result) + "\n")
            
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    args = parser()
    main(args)




    

    
    
