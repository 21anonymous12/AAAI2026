import sys
sys.path.append('./')
from copy import deepcopy
from utils.processor import get_custom_processor
import json
import pandas as pd
from io import StringIO

def preprocess_function(
    examples,
    tokenizer,
    max_source_length,
    max_target_length,
    ignore_pad_token_for_loss,
    padding,
    input_noise=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct", 
    tokenize=True,
):
    TABLE_PROCESSOR = get_custom_processor(
        max_cell_length=10000, 
        max_input_length=10000, 
        target_delimiter=', ',
        model_name=model_name
    )
    
    input_sources = []
    output_targets = []
    input_truncated = []
    
    for i in range(len(examples['question'])):
        question = examples['question'][i] 
        answer = examples['answer'][i] 
        table = pd.read_csv(StringIO(examples["table"][i]))
    
        table_content = {
            "header": table.columns.tolist(),
            "rows": table.values.tolist()
        }
        
        table_content_copy = deepcopy(table_content)
        
        input_source = TABLE_PROCESSOR.process_input(
            table_content_copy,
            question,
            [],
            True,
            False,
            False).strip().lower()
        
        input_sources.append(input_source)

        input_truncated.append(False)

        output_target = TABLE_PROCESSOR.process_output(answer).lower()
        output_targets.append(output_target)

    model_inputs = {}

    if tokenize:
        model_inputs = tokenizer(
            text=input_sources, 
            max_length=max_source_length, 
            padding=padding, 
            truncation=True
        )
    else:
        model_inputs['input_texts'] = input_sources
    
    labels = tokenizer(
        text=output_targets,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    
    if padding == "max_length" and ignore_pad_token_for_loss:
        pad_token_id = tokenizer.pad_token_id
        for label_ids in labels["input_ids"]:
            for j in range(len(label_ids)):
                if label_ids[j] == pad_token_id:
                    label_ids[j] = -100
    
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["truncated"] = [int(x) for x in input_truncated]
    
    return model_inputs