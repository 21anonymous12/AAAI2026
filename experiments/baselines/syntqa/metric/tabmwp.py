import re
import numpy as np
import pandas as pd
import json
import logging
from utils.metric import check_match
from utils.col_processor import col_processor
import sqlite3
import os

connection = sqlite3.connect(":memory:") 

def evaluate_sql_result(pred_string: str, gold_string: str, target_delimiter=', '):
    correct_flag = check_match(pred_string, gold_string)
    return correct_flag


def execute_sql_on_table(pred_sql: str, table_info: dict):
    columns = list(table_info.keys())
    rows = list(zip(*table_info.values()))  # 열 기준 -> 행 기준 변환

    table_content = {
        "header": columns,
        "rows": [list(r) for r in rows]
    }

    table_content["header"] = col_processor(table_content["header"])
    
    table_content["rows"] = [[str(cell).strip().lower() if isinstance(cell, str) else cell for cell in row] for row in table_content["rows"]]

    table = pd.DataFrame(table_content["rows"], columns=table_content["header"])
    
    try:
        table.to_sql('my_table', connection, index=False, if_exists="replace")  # 데이터프레임을 "data"라는 테이블로 저장
        # SQL 실행
        result = pd.read_sql_query(pred_sql, connection)
        # 결과를 셀 데이터만 추출해서 리스트로 변환
        results = result.values.tolist()
        result_str = ', '.join([str(item).strip().lower() for row in results for item in row])
        if not result_str:
            result_str = "none"
        error = ""
    except Exception as e:
        # 오류 발생 시 반환
        logging.warning(f"SQL 실행 실패: {e}")
        result_str = "none"
        error = str(e)
    
    return result_str, error


def postprocess_sql_predictions(decoded_preds, eval_dataset, fuzzy=False):
    L = len(decoded_preds)
    results = []  # SQL 실행 결과 string
    errors = []
    for i, pred_sql in enumerate(decoded_preds):
        # 간단 로깅
        _id = eval_dataset['id'][i]
        logging.warning(f'{i}/{L}: {str(_id)}')
        logging.warning(f'Raw SQL: {pred_sql}')

        table = eval_dataset['table_for_pd'][i]
        table = json.loads(table)

        result_str, error = execute_sql_on_table(pred_sql, table)

        results.append(result_str)
        errors.append(error)
        print('\n---------------------\n')
    return results, errors


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=False):
    def compute_metrics(eval_preds, meta=None, vllm=False):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        preds = []
        for pred in decoded_preds:
            try:
                matches = re.findall(r'"code"\s*:\s*"([^"]*)"', pred)
                match = matches[-1] if matches else None
                if match is None:
                    matches = re.findall(r"'code'\s*:\s*'''(.*?)'''", pred, re.DOTALL)
                    match = matches[-1] if matches else None
                
                if match is None:
                    pred = ""
                else:
                    pred = match.strip()
            except:
                pred = ""
            preds.append(pred)

        sql_results, errors = postprocess_sql_predictions(preds, eval_dataset, fuzzy)

        correct_flags = []
        for i, result_str in enumerate(sql_results):

            gold = eval_dataset["answer"][i]

            if isinstance(gold, list):
                gold_string = ", ".join([g.strip().lower() for g in gold])
            else:
                gold_string = str(gold).strip().lower()

            flag = evaluate_sql_result(result_str, gold_string, target_delimiter=', ')
            correct_flags.append(flag)

        acc = np.mean(correct_flags)

        if stage:
            to_save = {
                'id': eval_dataset['id'],
                'question': eval_dataset['question'],
                'answer': eval_dataset['answer'],  
                'acc': [int(b) for b in correct_flags],
                'sql_pred': preds,        
                'sql_result': sql_results,         
                'original_sql': decoded_preds,
                'error': errors,           
                'truncated': eval_dataset['truncated'],
                'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids']) if not vllm else eval_dataset['input_texts'],
            }
            if meta:
                to_save['log_probs_sum'] = meta.get('log_probs_sum', [])
                to_save['log_probs_avg'] = meta.get('log_probs_mean', [])

            try:  
                df = pd.DataFrame(to_save)
                os.makedirs(f'./predict/tabmwp', exist_ok=True)
                df.to_csv(f'./predict/tabmwp/{stage}.csv', na_rep='', index=False)
                print('predictions saved! (csv) ', stage)
            except Exception as e:
                logging.warning(f"CSV 저장 실패: {e}")
                with open(f'./predict/tabmwp/{stage}.json', 'w') as f:
                    json.dump(to_save, f)
                    print('predictions saved! (json) ', stage)

        return {"acc": np.round(acc, 4)}

    return compute_metrics
