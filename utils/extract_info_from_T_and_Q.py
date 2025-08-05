import json
from collections import Counter
import pandas as pd
import numpy as np
import re


def count_duplicate_words_in_headers(question: str, table_headers: list) -> int:
    
    
    question_cleaned = re.sub(r'[^\w\s]', '', question).lower()
    
    question_words = question_cleaned.split()
    
    
    header_words = []
    for header in table_headers:
        
        header_cleaned = re.sub(r'[^\w\s]', '', header).lower()
        header_words.extend(header_cleaned.split())
    
    
    question_word_set = set(question_words)
    header_word_set = set(header_words)
    
    
    common_words = question_word_set.intersection(header_word_set)
    return len(common_words)

def convert_to_numeric(table):
    

    
    original_columns = table.columns
    table.columns = [f"{col}_{i}" if list(table.columns).count(col) > 1 else col 
                     for i, col in enumerate(table.columns)]
    
    
    for col in table.select_dtypes(include='object').columns:
        numeric_col = pd.to_numeric(table[col], errors='coerce')
        
        table[col] = numeric_col.where(numeric_col.notna(), table[col])  
    
    
    table.columns = original_columns
    return table


def guess_column_type(dtype) -> str:
    
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DECIMAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    
    else:
        
        return "TEXT"


def extract_table_info(table, question):
     
    table_rows , table_columns  = table.shape
    table_size = table_rows * table_columns
    changed_table = convert_to_numeric(table)
    
    col_types = []
    for col in changed_table.columns:
        col_type = guess_column_type(changed_table[col].dtypes)
        col_types.append(col_type)
    
    int_check = False
    float_check = False
    bool_check = False
    text_check = False
    NaN_check = table.isna().any().any()
    if "INTEGER" in col_types:
        int_check = True
    
    if "DECIMAL" in col_types:
        float_check = True
    
    if "BOOLEAN" in col_types:
        bool_check = True

    if "TEXT" in col_types:
        text_check = True
    
    
    words = re.findall(r"[A-Za-z0-9]+", question)
    unique_words = set(word.lower() for word in words)
    unique_word_count = len(unique_words)
    
    
    numbers_in_text = re.findall(r"\d+", question)
    numbers_count = len(numbers_in_text)
    
    
    
    duplicate_count = count_duplicate_words_in_headers(question, table.columns.tolist())


    return {
        "table_row" : table_rows,
        "table_column" : table_columns,
        "table_size" : table_size,
        "table_int_check" : int_check,
        "table_float_check" : float_check,
        "table_bool_check" : bool_check,
        "table_text_check" : text_check,
        "table_NaN_check" : NaN_check,
        
        "question_unique_word_count" : unique_word_count,
        "question_numbers_count" : numbers_count,
        
        "table_question_duplicate_count" : duplicate_count
        
    }