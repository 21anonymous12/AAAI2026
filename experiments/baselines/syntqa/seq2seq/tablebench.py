import sys
sys.path.append('./')
from copy import deepcopy
from utils.processor import get_custom_processor
from utils.col_processor import col_processor
import json

def preprocess_function(examples,
                               tokenizer,
                               max_source_length,
                               max_target_length,
                               ignore_pad_token_for_loss,
                               padding,
                               input_noise=None,
                               model_name="meta-llama/Llama-3.2-3B-Instruct",
                               tokenize=True):

    TABLE_PROCESSOR = get_custom_processor(
        max_cell_length=10000,
        max_input_length=10000,
        target_delimiter=', ',
        model_name=model_name
    )

    inputs = []          
    outputs = []         
    input_truncated = [] 

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = examples["answer"][i]
        table = examples["table"][i]
    
        table_content = {
            "header": table['columns'],
            "rows": table['data']
        }

        table_content['header'] = col_processor(table_content['header'])

        table_content_copy = deepcopy(table_content)
        
        input_source = TABLE_PROCESSOR.process_input(
            table_content_copy,
            question,
            [],
            False,
            True,
            False).strip().lower()
                
        input_truncated.append(False)

        inputs.append(input_source)

        output = TABLE_PROCESSOR.process_output(answer).lower()
        outputs.append(output)
        
    model_inputs = {}

    if tokenize:
        model_inputs = tokenizer(
            text=inputs, 
            max_length=max_source_length, 
            padding=padding, 
            truncation=True
        )
    else:
        model_inputs['input_texts'] = inputs
    
    labels = tokenizer(
        text=outputs,
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
