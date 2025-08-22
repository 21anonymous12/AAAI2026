import os
import json

from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import Model
import numpy as np

from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes
from utils.table import transpose, sort_dataframe

from dataset.tabmwp import TabMWPDataset
from dataset.wikitq import WikiTQDataset
from dataset.tabfact import TabFactDataset
from dataset.fetaqa import FeTaQDataset
from dataset.tablebench import TableBenchDataset
from dataset.penguin import PenguinDataset
from dataset.general import GeneralDataset

from run_helper import get_cot_prompt, query, check_transpose, check_sort, read_json_file

def main(
        model:Optional[str] = "gpt-4o-mini", # base model of the agent (for short prompt to save money)
        provider: str = "openai", # openai, huggingface, vllm
        dataset:str = "wtq", # wtq or tabfact
        split:str = "test", # test, dev, train,
        perturbation: str = "none", # none, transpose, shuffle, transpose_shuffle
        norm: bool = True, # whether to NORM the table
        disable_resort: bool = True, # whether to disable the resort stage in NORM
        norm_cache: bool = True, # whether to cache the normalization results so that we can reuse them
        resume:int = 0, # resume from the i-th data point
        stop_at:int = 1e6, # stop at the i-th data point
        self_consistency:int = 10, # how many times to do self consistency
        temperature:float=0.8, # temperature for model
        log_dir: str = "output/wtq_cot", # directory to store the logs
        cache_dir: str = "cache", # directory to store the cache (normalization results)
):
    
    #### create log & cache dir and save config ####
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # store the config
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)
    
    #### load dataset and cot prompt ####
    if dataset == "wtq":
        data = WikiTQDataset(cache_dir='data', split=split)
    elif dataset == "tabfact":
        data = TabFactDataset(cache_dir='data', split=split)
    elif dataset == "tabmwp":
        data = TabMWPDataset(cache_dir='data', split=split)
    elif dataset == "fetaqa":
        data = FeTaQDataset(cache_dir='data', split=split)
    elif dataset == "tablebench":
        data = TableBenchDataset(cache_dir='data', split=split)
    elif dataset == "penguin":
        data = PenguinDataset(cache_dir='data', split=split)
    else:
        data = GeneralDataset(cache_dir='data', split=split)
    
    cot_prompt = get_cot_prompt(dataset)

    #### load the model ####
    if model:
        model = Model(model, provider=provider)
    # if long_model:
    #     long_model = Model(long_model, provider=provider)
        
    #### load the cache ####
    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))
    
    total = len(data)
    global_i = 0
    pbar = tqdm(total=stop_at if stop_at < total else total)

    for idx in range(total):
        if global_i < resume:
            global_i += 1
            pbar.update(1)
            continue
        if global_i >= stop_at:
            break
        
        d = data.get_item(idx)
        question = d["question"]
        answer = d["answer"]
        df = d["table"]
        table = df.to_markdown()
        question_id = d["question_id"]
        if isinstance(question_id, np.int64):
            question_id = int(question_id)
        
        title = d.get("title", "")
        table_id = d.get("table_id", idx)
        
        if norm:
            transpose_flag = False # check_transpose(model, model, table, title, table_id, perturbation, transpose_cache, norm_cache, cache_dir)
            
            if transpose_flag:
                transposed_df = transpose(df)
                df = remove_merged_suffixes(transposed_df)
            
            if not disable_resort:
                resort_list = check_sort(model, model, df, title, table_id, perturbation, resort_cache, norm_cache, cache_dir)
                df = sort_dataframe(df, resort_list)
        

        prompt = cot_prompt.replace("[TABLE]", table)\
            .replace("[QUESTION]", question)\
            .replace("[TITLE]", title)\
            .strip()

        text, response = query(model, model, prompt, temperature, self_consistency)
        
        log_path = os.path.join(log_dir, "log", f"{global_i}.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("===================Title===================\n")
            f.write(title + "\n")
            f.write("===================Table===================\n")
            f.write(table + "\n")
            f.write("===================Question===================\n")
            f.write(question + "\n")
            f.write("===================Text===================\n")
            f.write(text if isinstance(text, str) else "\n".join(text))
            f.write("\n")
            f.write("===================Answer===================\n")
            f.write(",".join(answer) if isinstance(answer, list) else str(answer))
            f.write("\n")
        
        res = {
            "idx": global_i,
            "answer": answer,
            "text": text,
            "table": table,
            "question": question,
            "question_id": question_id,
        }

        with open(os.path.join(log_dir, "result.jsonl"), "a") as f:
            json.dump(res, f)
            f.write("\n")

        global_i += 1
        pbar.update(1)

if __name__ == "__main__":
    Fire(main)