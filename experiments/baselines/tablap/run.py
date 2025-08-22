from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import json
from collections import Counter
from vllm import LLM, SamplingParams
import re
import gc 
import psutil, os
from tqdm import tqdm

def log_rss(tag: str=""):
    """현재 프로세스의 RSS(Resident Set Size)를 출력"""
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss  # 바이트 단위
    print(f"[{tag}] RSS: {rss / 1024**2:.2f} MB")

def log_gpu_memory(tag: str = ""):
    """
    torch.cuda 메모리 통계를 출력
    """
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA 사용 가능 GPU가 없습니다.")
        return

    # 현재 할당된 메모리
    allocated = torch.cuda.memory_allocated()  
    # 예약(reserved)된 메모리(캐시 포함)
    reserved  = torch.cuda.memory_reserved()    
    # 최대 할당 이력
    peak_alloc = torch.cuda.max_memory_allocated()  
    print(f"[{tag}] Allocated: {allocated/1024**2:.2f} MB, "
          f"Reserved: {reserved/1024**2:.2f} MB, "
          f"Peak Alloc: {peak_alloc/1024**2:.2f} MB")

"""
코드 절차 1에 해당하는 과정입니다. 일단 보내주신 mix-sc 결과에서 10개의 문제만 뽑았습니다.
"""

def load_predictions_from_jsonl(dp_path, agent_path, max_count=10, start_idx=0):
    results = []
    with open(dp_path, 'r') as f:
        dp_data = [json.loads(line) for line in f.readlines()]
    with open(agent_path, 'r') as f:
        agent_data = [json.loads(line) for line in f.readlines()]
    for i, (dp, agent) in enumerate(zip(dp_data, agent_data)):
        if i < start_idx:
            continue
        if i >= max_count + start_idx:
            break
        dp_preds = dp['preds']
        agent_preds = agent['preds']
        preds = dp_preds + agent_preds
        pred_count = Counter(preds)
        pred, _ = pred_count.most_common(1)[0]
        idx = preds.index(pred)
        table = dp['table']
        question = dp['question']
        answer = dp['answer']
        if isinstance(answer, list):
            answer = ", ".join(answer)
        reason = (dp['text'] + agent['text'])[idx]
        results.append([question, table, answer, pred, reason])
    return results

"""
코드 절차 2에 해당하는 과정입니다. 일단 주상님께서 보내주신 token으로 교체했습니다.
num solver를 셋업하고 선언하는 부분입니다.
"""

def setup_num_solver_pipeline(model_name: str):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name,
    #     trust_remote_code=True,
    # )
    # tokenizer.padding_side = "left"
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    # return pipeline(
    #     task="text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=512,
    #     pad_token_id=tokenizer.pad_token_id
    # )
    if 'gpt' in model_name:
        from openai import OpenAI
        client = OpenAI()
        
        return client
    else:
        # vllm
        model = LLM(
            model=model_name,
            tokenizer=model_name,
            tokenizer_mode="auto",
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.7,
            max_model_len=8192,
        )
        return model

"""
코드 절차 3에 해당하는 과정입니다. 여기도 주상님께서 보내주신 token으로 교체했습니다.
answer selector를 셋업하고 선언하는 부분입니다. 
"""

def setup_answer_selector_pipeline(model_name, adapter_path=None):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map='auto'
    )
    
    model.load_adapter(adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=3,
        batch_size=10,
        pad_token_id=tokenizer.pad_token_id
    )
    
def extract_reasoning(input_string):
    final_answer_index = input_string.find('```python')
    content = input_string[:final_answer_index].strip()
    return content

def extract_python_code(input_string):
    # Define the pattern to match Python code
    pattern = r'```python(.*?)```'

    # Use re.findall to find all matches
    matches = re.findall(pattern, input_string, re.DOTALL)

    # Return the matched Python code
    return matches[0]

def run_string(code_string):
    import subprocess
    try:
        # Run the script in a subprocess
        result = subprocess.run(['python', '-c', code_string], capture_output=True, timeout=30, text=True, check=True)

        # Print the captured output
        return(result.stdout.strip())

    except subprocess.CalledProcessError as e:
        return(f"An error occurred: {e}")
    
def filter_answer(raw_context) ->str:
    pattern = r'Final Answer: (.*)'
    result = re.findall(pattern, raw_context)
    if result:
        result = result[-1].strip()
    else:
        result = ""
    return result
    
def get_refine_answer(respon):
    org_propose = respon
    instruct = extract_reasoning(respon)
    instruction = instruct
    refine_propose = ""
    python_code = ""
    filter_ans = ""
    python_res = ""
    
    if "python" in respon:
        try:
            python_code = extract_python_code(respon)
            python_res = run_string(python_code)
            if "Final Answer" in python_res and "error" not in python_res:
                python_res = filter_answer(python_res)
        except:
            python_res = ""
            respon = respon.replace(python_code, "")
    if "Final Answer" in respon:
        filter_ans = filter_answer(respon)

    if not filter_ans:
        refine_propose = python_res
    elif not python_res:
        refine_propose = filter_ans
    elif filter_ans and python_res:
        if "error" in python_res:
            refine_propose = filter_ans
        else:
            refine_propose = python_res
    else:
        refine_propose = "no result"
        
    return refine_propose, instruction, org_propose

"""
각 num solver와 answer selector prompt에 해당하는 부분입니다.
"""

# Answer Selector 프롬프트 구성
ANS_SELECTOR_PROMPT = """Below is a table header regarding [TABLE_TITLE]:
Header: [TABLE_HEADER]
You're tasked with answering the following question:
[QUESTION]
You have 2 answers with their corresponding reasoning processes derived by two different models. Answer [A] was derived by the A model. Answer [B] was derived by B model.
Reasoning of Answer [A] is: [REASON_A]
Answer [A] is: [ANSWER_A].
Reasoning of Answer [B] is: [REASON_B]
Answer [B] is: [ANSWER_B].
Your task is to determine which is the correct answer. The final answer is [A] if Answer A is correct, and [B] if Answer B is correct.
If Answer [A] and Answer [B] are the same, the final answer could be either [A] or [B].
Therefore, the final answer is:
"""

ANS_SELECTOR_PROMPT_TABFACT = """
Below is a table header regarding [TITLE]:
Header: [HEADER]
You're tasked with determing whether the given statement is True:
[STAT]
You have 2 answers with their corresponding reasoning processes derived by two different models. Answer [A] was derived by the A model. Answer [B] was derived by B model.
For the answers, yes means True and no means False
Reasoning of Answer [A] is: [RES1]
Answer [A] is: [ANS1]
Reasoning of Answer [B] is: [RES2]
Answer [B] is: [ANS2]
Your task is to determine which is the correct answer. The final answer is [A] if Answer A is correct, and [B] if Answer B is correct.
If Answer [A] and Answer [B] are the same, the final answer could be either [A] or [B].
Therefore, the final answer is:
"""

# Num Solver 프롬프트 구성
PROMPT_MATH_SOLVER = """
Please answer the question according to the given table regarding [TITLE] 
The table context is: [TABLE]
The question is: [QUESTION]

Notes:
- Try to solve the problem step by step and give the process of deducing the answer with intermediate results (as concise as possible)
- Answer the question according to the columns/rows which contexts are most related to the question context
- Give me the answer in format "Final Answer: AnswerName1, AnswerName2..." form (should be a number or entity names, as short as possible, without any explanation)
- Meanwhile, give me the python script (prefer using list operations instead of dataframe)
- Use print function to output the final answer (note: do not add any extra context in the print)
- If python contains subtraction, use absolute values
- For the answer, keep only two decimal places
"""

PROMPT_MATH_SOLVER_TABFACT = """
Please determine whether a given statement is true or not, according to the given table regarding [TITLE]
The table header is: [HEADER]
The table context is: [TABLE]
The statement is: [STAT]
Notes:
- Try to solve the problem step by step and give the reasoning process firstly
- Then, give me the python script
- Use print function to output the final answer (note: do not add any extra context in the print)
- Give me the answer in format "Final Answer: Answer" form (either True or False, without any explanation)
- Directly output the answer after the python script
"""

def get_num_solver_prompt(question, table, template):
    if '[STAT]' in template:
        prompt = template.replace("[TITLE]", "")
        prompt = prompt.replace("[HEADER]", table.splitlines()[0])
        prompt = prompt.replace("[TABLE]", table)
        prompt = prompt.replace("[STAT]", question)
    else:
        prompt = template.replace(" [TITLE]", "")
        prompt = prompt.replace("[TABLE]", table)
        prompt = prompt.replace("[QUESTION]", question)
    return prompt

def get_answer_selector_prompt(question, table, reason_a, answer_a, reason_b, answer_b, template):
    prompt = template
    prompt = prompt.replace(" [TABLE_TITLE]", "")
    prompt = prompt.replace("[TABLE_HEADER]", table.splitlines()[0])
    prompt = prompt.replace("[QUESTION]", question)
    prompt = prompt.replace("[REASON_A]", reason_a)
    prompt = prompt.replace("[ANSWER_A]", answer_a)
    prompt = prompt.replace("[REASON_B]", reason_b)
    prompt = prompt.replace("[ANSWER_B]", answer_b)
    return prompt

def parser():
    import argparse
    parser = argparse.ArgumentParser(description="Run the model")
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3.2-3b-instruct", help="Model name")
    parser.add_argument("--adapter_path", type=str, default="./example_ckpt", help="Adapter path")
    parser.add_argument("--max_count", type=int, default=10, help="Max count of predictions to load")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for predictions")
    parser.add_argument("--dp", type=str, default="wtq_dp_llama3.2_3b_result.jsonl", help="File 1")
    parser.add_argument("--agent", type=str, default="wtq_agent_llama3.2_3b_result.jsonl", help="File 2")
    parser.add_argument("--output", type=str, default="wtq_llama3.2_3b_result.jsonl", help="Output file")
    parser.add_argument("--dataset", type=str, default="wtq", help="Dataset name")
    return parser.parse_args()

def main(args):
    mix_sc_preds = load_predictions_from_jsonl(args.dp, args.agent, max_count=args.max_count, start_idx=args.start_idx)

    num_solver = setup_num_solver_pipeline(model_name=args.model_name)
    
    if not 'gpt' in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    results = []

    import time
    start_time = time.time()
    
    for i in tqdm(range(len(mix_sc_preds))):
        idx, (question, table, answer, pred, reason) = i, mix_sc_preds[i]

        log_rss("Before processing")
        log_gpu_memory("Before processing")

        # Clear cache every 100 iterations
        if i > 0 and i % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Cleared cache")
        
        # Generate the num solver prompt
        if args.dataset == "tabfact":
            num_solver_prompt = get_num_solver_prompt(question, table, PROMPT_MATH_SOLVER_TABFACT)
        else:
            num_solver_prompt = get_num_solver_prompt(question, table, PROMPT_MATH_SOLVER)

        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=2024,
        )

        if 'gpt' in args.model_name:
            input = num_solver_prompt
        else:
            input = tokenizer.apply_chat_template(
                [{"role": "user", "content": num_solver_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        
        if args.dataset == "tabfact":
            num_solver_response = num_solver.generate(input, sampling_params)[0].outputs[0].text
            num_final_answer, num_solver_reason, _ = get_refine_answer(num_solver_response)
            if "false" in num_final_answer.lower():
                num_final_answer = "no"
            else: 
                num_final_answer = "yes"
        else:
            if 'gpt' in args.model_name:
                num_solver_response = num_solver.chat.completions.create(
                    model='gpt-4o',
                    messages=[{"role": "user", "content": input}],
                    temperature=0.1,
                    max_tokens=2024,
                ).choices[0].message.content
                num_final_answer = num_solver_response.split("Final Answer:")[-1].split("\n")[0]
                num_solver_reason = num_solver_response.split("```")[-1]
            else:
                num_solver_response = num_solver.generate(input, sampling_params)[0].outputs[0].text
                num_final_answer = num_solver_response.split("Final Answer:")[-1].split("\n")[0]
                num_solver_reason = num_solver_response.split("```")[-1]

        if pred is None:
            pred = ""
        if num_final_answer is None:
            num_final_answer = ""

        ans_selector_prompt = get_answer_selector_prompt(
            question,
            table,
            reason,
            pred,
            num_solver_reason,
            num_final_answer,
            ANS_SELECTOR_PROMPT
        )

        results.append({
            "idx": idx,
            "question": question,
            "table": table,
            "answer": answer,
            "pred": pred,
            "reason": reason,
            "num_solver_response": num_solver_response,
            "num_final_answer": num_final_answer,
            "ans_selector_prompt": ans_selector_prompt,
        })

    start_time = time.time()
    print("Setting up answer selector pipeline...")

    # clear cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Cleared cache")

    classifier = setup_answer_selector_pipeline(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_path=args.adapter_path,
    )

    for i, result in enumerate(results):
        print(f"Processing {i+1}/{len(results)}, time elapsed: {time.time() - start_time:.2f}s")
        idx = result["idx"]
        question = result["question"]
        table = result["table"]
        answer = result["answer"]
        pred = result["pred"]
        reason = result["reason"]
        num_solver_response = result["num_solver_response"]
        num_final_answer = result["num_final_answer"]
        ans_selector_prompt = result["ans_selector_prompt"]

        selector_response = classifier(ans_selector_prompt)[0]['generated_text']
        selector_response = selector_response[len(ans_selector_prompt):].strip()
        
        if '[A]' in selector_response:
            final_answer = pred
        elif '[B]' in selector_response:
            final_answer = num_final_answer
        else:
            final_answer = pred

        results[i]["selector_response"] = selector_response
        results[i]["final_answer"] = final_answer

    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print("Results saved to ", args.output)
        

if __name__ == "__main__":
    args = parser()
    main(args)
    # results = load_predictions_from_jsonl("wtq_dp_llama3.2_3b_result.jsonl", "wtq_agent_llama3.2_3b_result.jsonl", max_count=10)
    # print("Results:", results)