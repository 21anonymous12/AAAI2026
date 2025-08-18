import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, load_dataset
import pandas as pd
import torch
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from utils.create_query_format import *
from langchain_core.prompts import load_prompt
from utils.run_pandas_code import *
from utils.LLM_inference_with_scheduler import *
from utils.LLM_inference_without_scheduler import *
from utils.selector_inference import *
from utils.convert_table_datatype import *
from utils.run_SQL_code import *
from utils.make_unique_columns import *
from utils.convert_selector_format import *
from utils.long_text_check import *
from utils.choose_ans_based_on_selector import *
from utils.extract_info_from_T_and_Q import *
from utils.JA_extract_answer import *
from utils.FM_inference import *
from scheduler.scheduler import MobileBertWithFeatures
from langchain.tools import tool
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
import argparse
import yaml
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
parser.add_argument('--Use_Scheduler', type=lambda x: x.lower() == 'true', help='Override Use_Scheduler')
parser.add_argument('--N', type=int, help='Override N')
parser.add_argument('--JA_threshold', type=float, help='Override JA_threshold')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
if args.Use_Scheduler is not None:
    config['Use_Scheduler'] = args.Use_Scheduler
if args.N is not None:
    config['N'] = args.N
if args.JA_threshold is not None:
    config['JA_threshold'] = args.JA_threshold

Use_Scheduler = config['Use_Scheduler']
N = config['N']
JA_threshold = config['JA_threshold']


from Tablebench_loader import Tablebench_Filtered

dataset = load_dataset(
    path="Multilingual-Multimodal-NLP/TableBench",
    data_files="TableBench.jsonl",
    split="train"
)


prompt_PoT = load_prompt("prompt/PoT_prompt.yaml", encoding="utf-8")
prompt_text2sql = load_prompt("prompt/text2sql_prompt.yaml", encoding="utf-8")
prompt_CoT = load_prompt("prompt/CoT_prompt.yaml", encoding="utf-8")
prompt_refine_PoT = load_prompt("prompt/PoT_self-refine.yaml", encoding="utf-8")
prompt_refine_text2sql = load_prompt("prompt/text2sql_self-refine.yaml", encoding="utf-8")
prompt_FM = load_prompt("prompt/core_extract_prompt.yaml", encoding="utf-8")


confidence_checker_path = "7anonymous7/confidence_checker"
confidence_checker_tokenizer = AutoTokenizer.from_pretrained(confidence_checker_path)
confidence_checker = AutoModelForSequenceClassification.from_pretrained(confidence_checker_path)
confidence_checker.eval()

confidence_checker.to(device)

# 도구 정의
@tool
def check_confidence_scores(question: str) -> str:
    """Output the confidence scores for three types of reasoning on the question about the table: text-based reasoning (CoT), reasoning using pandas (PoT), and reasoning using SQL (textSQL)."""
    if results_all.get("inference") == "inference error":
        test_data = "inference error"
    test_data = transform_json_to_special_tokens(table, question, results_all, window_number = 0)
    
    if test_data == "inference error":
        print("An error occurred during the LLM inference process.")
    long_checked_text_data = long_text_check(test_data, confidence_checker_tokenizer, check_PoT_error=check_PoT_error, check_text2sql_error=check_text2sql_error)
    
    confidence_checker_result = predict_label(long_checked_text_data, confidence_checker_tokenizer, confidence_checker)
    print("Tool used for confidence scores")
    confidence_scores = f'reasoning using pandas (PoT) : {confidence_checker_result[0]}\nreasoning using SQL (textSQL) : {confidence_checker_result[1]}\ntext-based reasoning (CoT) : {confidence_checker_result[2]}'
    return confidence_scores

# tools 정의
tools = [check_confidence_scores]

tool_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI that provides appropriate answers to users’ questions."
            "Your task is to output both the Answer to a Question about a given Table and the Justification for that Answer."
            "The provided Table is as follows:"
            "{table}"
            "The Question about the Table is as follows:"
            "{question}"
            "The user will also provide 'the results of three different reasoning approaches performed by another AI to answer the Question about the Table.' You should use these as references to produce the optimal Answer."
            "These reasoning results will be provided in JSON format. The three reasoning approaches are detailed below:"
            "1. reasoning using pandas (PoT): This refers to reasoning about the Question using Python’s pandas library, including the generated code and its execution results. Here, N indicates the number of times the code was refined. A larger N means multiple refinements were attempted. For example, code with N=1 was obtained by refining the code with N=0."
            "2. reasoning using SQL (textSQL): This refers to reasoning about the Question using SQL, including the generated code and its execution results. As with PoT, N indicates the number of refinement iterations."
            "3. Text-based reasoning (CoT): This refers to reasoning about the Question purely in natural language and its resulting answer. CoT does not involve any refinements; therefore, N is always 0."
            "You must use the provided reasoning results to determine the best possible final answer."
            "Make sure to use the `check_confidence_scores` tool for obtaining confidence scores for each approach."
            "If these results alone are insufficient or ambiguous, you may optionally use the `check_confidence_scores` tool to obtain confidence scores for each reasoning result. However, do not rely on these confidence scores as 100% accurate—they are only meant for reference. You may choose whether or not to use this tool."
            "Return the final output strictly in the following JSON format.: {{'Justification': 'To calculate the rate of change, subtract the number of boxes sold on Wednesday (27) from Thursday (23): 23 - 27 = -4. Since the change occurred over 1 day, the rate of change is -4 boxes per day.',  'Answer': '-4'}}",
        ),
        ("human", "{results_all}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

tablebench_question_list = []
tablebench_pred_list = []
tablebench_answer_list = []
for index in tqdm(range(693)):
    llm = ChatOllama(
        model="qwen2.5:32b-instruct", 
        format="json",  
        temperature=0,
        num_ctx = 2048,  
    )
    
    Tablebench = Tablebench_Filtered(dataset, index) # index는 692까지 있음. 693부터 out of index
    Tablebench_table = Tablebench.get_table()
    question = Tablebench.get_question()
    Tablebench_answer = Tablebench.get_answer()
    Tablebench_qsubtype = Tablebench.get_qsubtype()
    
    table = make_unique_columns(Tablebench_table) # If column names are duplicated, this function add additional numbers at columns. (ex. [Day, Month, Day, Month] => [Day 1, Month 1, Day 2, Month 2]) 
    
    results_all = {}
    PoT_results = {}
    text2sql_results = {}
    CoT_results = {}
    
    
    
    if Use_Scheduler == True :
        num_feats_cols = ['table_row', 'table_column', 'table_size', 'question_unique_word_count', 'question_numbers_count', 'table_question_duplicate_count']
        bool_feats_cols = ['table_int_check', 'table_float_check', 'table_text_check', 'table_NaN_check']
        num_extra = len(num_feats_cols) + len(bool_feats_cols)
        info = extract_table_info(table, question)
        scheduler_tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
        scheduler = MobileBertWithFeatures(num_extra).to(device)
        scheduler.load_state_dict(torch.load("./scheduler/mobilebert_multilabel_45.pt", map_location=device))
        scheduler.eval()
        sample = {}
        sample["sentence"] = question + " | " + str(table.columns.tolist())
        sample["num_feats"] = [info[key] for key in num_feats_cols]
        sample["bool_feats"] = [info[key] for key in bool_feats_cols]
    
        enc = scheduler_tokenizer(
            sample["sentence"],
            padding="max_length",
            truncation=True,
            max_length=500,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
    
        extra_feats = torch.tensor(
            sample["num_feats"] + [float(b) for b in sample["bool_feats"]],
            dtype=torch.float32
        ).unsqueeze(0).to(device) 
        with torch.no_grad():
            logits = scheduler(input_ids, attention_mask, extra_feats)  # [1, 2]
            probs = torch.sigmoid(logits).cpu().numpy() 
        if probs[0][0] >= probs[0][1]:
            scheduler_result = 'PoT'    
        elif probs[0][0] < probs[0][1]:
            scheduler_result = 'text2sql'  
        PoT_results, text2sql_results, CoT_results = LLM_inference_with_scheduler(prompt_PoT, prompt_text2sql, prompt_CoT, prompt_refine_PoT, prompt_refine_text2sql, llm, table, question, PoT_results, text2sql_results, CoT_results, scheduler_result , N = N)
    elif Use_Scheduler == False :
        PoT_results, text2sql_results, CoT_results = LLM_inference_without_scheduler(prompt_PoT, prompt_text2sql, prompt_CoT, prompt_refine_PoT, prompt_refine_text2sql, llm, table, question, PoT_results, text2sql_results, CoT_results, N = N)
    
    results_all = {
        'PoT': PoT_results,
        'text2sql' : text2sql_results,
        'CoT': CoT_results
    }
    # Grond Truth : -4
    
    # with open('LLM_inference_results.json', 'w', encoding='utf-8') as json_file:
    #     json.dump(results_all, json_file, ensure_ascii=False, indent=4)
    
    
    
    
    if results_all.get("inference") == "inference error":
        test_data = "inference error"
    test_data = transform_json_to_special_tokens(table, question, results_all, window_number = 0)
    
    if test_data == "inference error":
        print("An error occurred during the LLM inference process.")
    long_checked_text_data = long_text_check(test_data, confidence_checker_tokenizer, check_PoT_error=check_PoT_error, check_text2sql_error=check_text2sql_error)
    
    confidence_checker_result = predict_label(long_checked_text_data, confidence_checker_tokenizer, confidence_checker)
    
    selected_answer, NOT_judge_agent_call = choose_ans_based_on_selector(confidence_checker_result, results_all, JA_threshold = JA_threshold)
    
    
    
    if NOT_judge_agent_call:
        print("================================CC used================================")
        final_answer = selected_answer
    
    else:
        llm2 = ChatOllama(
            model="qwen2.5:32b-instruct", 
            format="json",  
            temperature=0,
            num_ctx = 2048,  
            )
        
        token_count_result = measure_and_adjust_context(tool_prompt.format(table = table.to_markdown(index=False), question = question, results_all = results_all))
        if token_count_result['adjusted_context'] > 2048:
            llm2 = llm_adjusted_context(llm2, token_count_result['adjusted_context'])
            print(llm2)
        
        agent = create_tool_calling_agent(llm2, tools, tool_prompt)
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )
    
        agent_result = agent_executor.invoke({"table":table.to_markdown(index=False), "question": question, "results_all": results_all})
    
        final_answer = extract_answer_from_JA_output(agent_result)
        print("================================JA used================================")
    
    if isinstance(final_answer, str) and len(final_answer) > 100:
        print("================================FM used================================")
        final_answer = FM_inference(prompt_FM, question, final_answer)
    
    
    print("================================FINAL ANSWER================================")
    print(f"{index} Answer : {final_answer}")

    
    tablebench_question_list.append(question)
    tablebench_pred_list.append(final_answer)
    tablebench_answer_list.append(Tablebench_answer)

# DataFrame 생성
df = pd.DataFrame({
    "Question": tablebench_question_list,
    "Prediction": tablebench_pred_list,
    "Answer": tablebench_answer_list
})

# CSV 저장
df.to_csv("MATA_tablebench_qwen2-5_32b.csv", index=False, encoding="utf-8-sig")

print("CSV saved!")