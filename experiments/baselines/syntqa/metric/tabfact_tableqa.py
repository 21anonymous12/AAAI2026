import numpy as np
import pandas as pd
import logging
from utils.metric import check_match
from utils.eval import extract_answer
import json
import os

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = ['yes' if label==1 else 'no' for label in labels]
    return preds, labels

def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None): 
    def compute_metrics(eval_preds, meta=None, vllm=False):
        # nonlocal tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        preds = []
        for pred in decoded_preds:
            try:
                pred = extract_answer(pred)
                if pred is None:
                    pred = ""
                preds.append(pred)
            except:
                preds.append("")

        preds, labels = postprocess_text(preds, eval_dataset['label'])

        correct_flag = []
        for i, (pred, label) in enumerate(zip(preds, labels)):
            correct_flag.append(check_match(pred, label))

        if stage:
            to_save = {'id': eval_dataset['id'],
                       'question': ["Based on given table, check the following statement is true or false. Statement: " + statement for statement in eval_dataset['statement']],
                       'answer': labels,
                       'acc': [int(b) for b in correct_flag],
                       'predictions': preds,
                       'truncated': eval_dataset['truncated'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids']) if not vllm else eval_dataset['input_texts']}
            if meta:
                to_save['log_probs_sum'] = meta['log_probs_sum']
                to_save['log_probs_avg'] = meta['log_probs_mean']

            try:
                df = pd.DataFrame(to_save)
                os.makedirs(f'./predict/tabfact', exist_ok=True)
                df.to_csv(f'./predict/tabfact/{stage}.csv', na_rep='',index=False)
                print('predictions saved! ', stage)
            except Exception as e:
                logging.warning(f"CSV 저장 실패: {e}")
                with open(f'./predict/tabfact/{stage}.json', 'w') as f:
                    json.dump(to_save, f)
                    print('predictions saved! (json) ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics


