import json
from collections import Counter
import pandas as pd
import numpy as np
import re
import ast
from tqdm import tqdm


error_PoT_pattern = ["Can only use .str accessor with string values!", ".str accessor", "unsupported operand", "Reindexing only valid", "Series is ambiguous", "arg must be", "string values!", "Non-numeric values", "Non-numeric", "<NOTHING>","nameerror","typeerror","<unknown>","keyerror","syntaxerror","indexerror", "valueerror", "dtype='int64'", "no overlapping index names", "cannot reindex", "duplicate labels", "bytes-like object", "is not defined", "argument must be a string or a real number", "'NAType'", "first argument must be an iterable of pandas objects", "you passed an object of type", "Error: ", "cannot be concatenated", "cannot join with no overlapping index names","must be of the same length","does not match format","scalar, tuple, or list","NoneType","must be a string","None of [Index(", "invalid literal for ", "out of bounds for", "has no attribute", "invalid syntax", "(<string>, line", "dtype: object", "not supported between instances", "list index out of range",  "is out-of-bounds", "unhashable type:", "None of [Index", "invalid index", "index out of range", "not convert string to float", "not found in axis", "The truth value of a Series is ambiguous.", "Length mismatch: Expected ", "get argmax of an empty sequence", "Unable to parse string", 'arg is an empty sequence', 'No unique match found', '(expected ', 'not enough values to unpack', 'expected hh:mm:ss format before', ' object is not iterable', "doesn't match format", "dtype: int64", "can only concatenate str", "unsupported operand type(s) for", "pattern contains no capture groups", "nothing to repeat at position", "takes no keyword arguments", "Something has gone wrong, please report a bug at", "No valid entry dates", "attempt to get argmin", "No specific type mentioned", "column not found in data", "arg must be a list, tuple, 1-d array, or Series", "did not contain a loop with signature matching types", "<class ", "cannot convert float", "nomination found for", "No valid attendance data available", "Length of values", "not match length of index", "Invalid comparison between", "Data not available in the provided DataFrame", "unexpected keyword argument", "not supported for the", "No specific issue range found for", "does not support strftime", "No valid", "unconverted data remains when parsing with format", "could not be broadcast", "operands could not", "object do not match", "Unalignable boolean Series", "Boolean array expected", "Series.replace cannot use dict-like to_replace and non-None value", "Expected value of kwarg", "No matching driver combination found", "unit must not be specified", "if the input contains a str", "integer or boolean arrays are valid indices"]
error_PoT_match_pattern = ["<NOTHING>","code", "'code'", "'index'", "'ans'", "nan", "None", "[nan]", "ans", "$nan", "[]"]


def check_PoT_error(text):
    if len(text) > 3000:
        return False
    for pattern in error_PoT_pattern:
        if pattern in text:
            return False
    for pattern in error_PoT_match_pattern:
        if text == pattern:
            return False
    return text

error_text2sql_pattern = ["<NOTHING>",'Error:', '[nan]']
error_text2sql_match_pattern = ["<NOTHING>", "code", "'code'", '[]']


def check_text2sql_error(text):
    if len(text) > 3000:
        return False
    for pattern in error_text2sql_pattern:
        if pattern in text:
            return False
    for pattern in error_text2sql_match_pattern:
        if text == pattern:
            return False
    return text



def choose_answer(answer_list, reasoning_json):
    methods = pd.Series({
    'PoT': answer_list[0],
    'text2SQL': answer_list[1],
    'CoT': answer_list[2]
    }).sort_values(ascending=False).index.tolist()

    for method in methods:
        pot_execution_results = []
        text2sql_execution_results = []
        if method == 'CoT':
            try:
                return str(reasoning_json['CoT']['N=0']['answer']).strip("'").strip('"')
            except (KeyError, TypeError):
                
                continue

        elif method == 'PoT':
            pot_data = reasoning_json['PoT']
            pot_execution_results = [
                pot_data[key] for key in sorted(pot_data.keys())
                if key.startswith("N=") and key.endswith("execution_result")
            ]
            for i in range(len(pot_execution_results)):
                if not check_PoT_error(pot_execution_results[i]):
                    pot_execution_results[i] = "玄玄玄"
            pot_execution_results = [s for s in pot_execution_results if '玄' not in s]
            if len(pot_execution_results) != 0:
                return pot_execution_results[-1].replace("[", "").replace("]", "").strip("'").strip('"')
            elif len(pot_execution_results) == 0:
                continue 

        elif method == 'text2SQL':
            text2sql_data = reasoning_json['text2sql']
            text2sql_execution_results = [
                text2sql_data[key] for key in sorted(text2sql_data.keys())
                if key.startswith("N=") and key.endswith("execution_result")
            ]
            for i in range(len(text2sql_execution_results)):
                if not check_text2sql_error(text2sql_execution_results[i]):
                    text2sql_execution_results[i] = "玄玄玄玄"
            text2sql_execution_results = [s for s in text2sql_execution_results if '玄' not in s]
            if len(text2sql_execution_results) != 0:
                return text2sql_execution_results[-1].replace("[", "").replace("]", "").strip("'").strip('"')
            elif len(text2sql_execution_results) == 0:
                continue 


def choose_ans_based_on_selector(scheduler_output, result_json, JA_threshold = 0.1):

    NOT_judge_agent_call = not all(x < JA_threshold for x in scheduler_output)

    try:
        selected_answer = choose_answer(scheduler_output, result_json)
    except:
        print("ERROR occured!")


    return selected_answer, NOT_judge_agent_call