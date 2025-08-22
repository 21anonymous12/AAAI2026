# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .table_custom_linearize import IndexedRowTableLinearize as CustomIndexedRowTableLinearize
from .table_custom_truncate import CellLimitTruncate as CustomCellLimitTruncate, RowDeleteTruncate as CustomRowDeleteTruncate
from .table_linearize import IndexedRowTableLinearize
from .table_truncate import CellLimitTruncate, RowDeleteTruncate
from .table_processor import TableProcessor
from .table_custom_processor import TableProcessor as CustomTableProcessor
from transformers import AutoTokenizer


def get_custom_processor(max_cell_length, max_input_length, target_delimiter=', ', model_name="meta-llama/Llama-3.2-3B-Instruct"):
    table_linearize_func = CustomIndexedRowTableLinearize()
    table_truncate_funcs = [
        CustomCellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name), 
                          max_input_length=max_input_length),
        CustomRowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = CustomTableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs,
                               target_delimiter=target_delimiter, 
                               tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name))
    return processor

def get_default_processor(max_cell_length, max_input_length, target_delimiter=', '):
    table_linearize_func = IndexedRowTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large"),
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs,
                               target_delimiter=target_delimiter)
    return processor
