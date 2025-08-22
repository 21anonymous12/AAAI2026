import os
import json
from typing import Optional
from tqdm import tqdm
from fire import Fire
from agent import TableAgent, Model
from utils.data import construct_markdown_table
from utils.execute import markdown_to_df, remove_merged_suffixes, convert_cells_to_numbers
from utils.table import transpose, sort_dataframe
from run_helper import load_dataset, check_transpose, check_sort, read_json_file
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_table(
        d, 
        model, 
        long_model, 
        dataset, 
        perturbation, 
        norm, 
        disable_resort, 
        norm_cache, 
        transpose_cache, 
        resort_cache, 
        cache_dir, 
        use_full_table, 
        self_consistency, 
        temperature, 
        log_dir):

    results = []
    table_id = d["table_id"]
    title = d["title"]

    if perturbation == "none":
        table = construct_markdown_table(**d["table"])
    elif perturbation == "transpose":
        table = construct_markdown_table(**d["transposed_table"])
    elif perturbation == "shuffle":
        table = construct_markdown_table(**d["row_shuffled_table"])
    elif perturbation == "transpose_shuffle":
        table = construct_markdown_table(**d["row_shuffled_transposed_table"])

    df = markdown_to_df(table)

    transpose_flag = False
    resort_list = []

    if norm:
        transpose_flag = check_transpose(model, long_model, table, title, table_id, perturbation, transpose_cache, norm_cache, cache_dir)
        if transpose_flag:
            transposed_df = transpose(df)
            df = remove_merged_suffixes(transposed_df)
        if not disable_resort:
            resort_list = check_sort(model, long_model, df, title, table_id, perturbation, resort_cache, norm_cache, cache_dir)
            df = sort_dataframe(df, resort_list)

    df = convert_cells_to_numbers(df)
    table = df.to_markdown()

    # Ensure log directory exists for table_id
    table_log_dir = os.path.join(log_dir, str(table_id))
    os.makedirs(table_log_dir, exist_ok=True)

    for idx, question in enumerate(d["questions"]):
        answer = d["answers"][idx]
        question_id = d["ids"][idx]

        texts = []
        log_file_path = os.path.join(table_log_dir, f"{idx}.log")

        for _ in range(self_consistency):
            agent = TableAgent(
                table=df,
                prompt_type=dataset,
                model=model,
                long_model=long_model,
                temperature=temperature,
                log_dir=log_file_path,
                use_full_table=use_full_table,
            )
            text, _ = agent.run(question=question, title=title)
            texts.append(text)

        res = {
            "answer": answer,
            "text": texts if self_consistency > 1 else texts[0],
            "transpose": transpose_flag,
            "resort": resort_list,
            "question_id": question_id,
            "table_id": table_id,
            "title": title,
            "table": table,
            "question": question,
        }
        results.append(res)
        print(f"Table {table_id}, Question {question_id} processed")
    return results

def main(
        model:Optional[str] = "gpt-3.5-turbo-0613",
        long_model:Optional[str] = "gpt-3.5-turbo-16k-0613",
        provider: str = "openai",
        dataset:str = "wtq",
        perturbation: str = "none",
        use_full_table: bool = True,
        norm: bool = True,
        disable_resort: bool = True,
        norm_cache: bool = True,
        sub_sample: bool = True,
        resume:int = 0,
        stop_at:int = 1e6,
        self_consistency:int = 1,
        temperature:float=0.8,
        log_dir: str = "output/tabfact_agent",
        cache_dir: str = "cache",
        num_workers: int = 4
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({key: value for key, value in locals().items() if key != 'f'}, f, indent=4)

    data = load_dataset(dataset)
    model = Model(model, provider=provider)
    long_model = Model(long_model, provider=provider)

    transpose_cache = read_json_file(os.path.join(cache_dir, "transpose.json"))
    resort_cache = read_json_file(os.path.join(cache_dir, "resort.json"))

    total = sum([len(d['sampled_indices']) for d in data]) if sub_sample else sum([len(d['questions']) for d in data])
    pbar = tqdm(total=stop_at if stop_at < total else total)

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_table = {executor.submit(
            process_table,
            d, model, long_model, dataset, perturbation, norm, disable_resort, norm_cache, transpose_cache,
            resort_cache, cache_dir, use_full_table, self_consistency, temperature, log_dir): d for d in data}

        for future in as_completed(future_to_table):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                print(f"Error processing table: {e}")
                
            pbar.update(len(result))

    with open(os.path.join(log_dir, "result.jsonl"), "w") as f:
        for res in results:
            json.dump(res, f)
            f.write("\n")

if __name__ == "__main__":
    Fire(main)
