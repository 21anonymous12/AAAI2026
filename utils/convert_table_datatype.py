import pandas as pd
import numpy as np

def convert_to_numeric(table):
    original_columns = table.columns
    table.columns = [f"{col}_{i}" if list(table.columns).count(col) > 1 else col 
                     for i, col in enumerate(table.columns)]
    
    
    for col in table.select_dtypes(include='object').columns:
        numeric_col = pd.to_numeric(table[col], errors='coerce')
        
        table[col] = numeric_col.where(numeric_col.notna(), table[col])  
    
    
    table.columns = original_columns
    return table


def guess_sql_type(dtype) -> str:

    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DECIMAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    
    else:
        
        return "TEXT"

def df_to_table_prompt(df: pd.DataFrame, table_name: str = "dataframe") -> str:
    
    
    lines = [f"-- Table: {table_name}"]
    lines.append("-- Columns:")
    
    
    for col in df.columns:
        col_type = guess_sql_type(df[col].dtypes)
        lines.append(f"--   {col} ({col_type})")

    
    lines.append("--")
    lines.append("-- Rows:")
    
    for idx, row in df.iterrows():
        
        row_values = [str(val) for val in row.values]
        lines.append(f"--   {' | '.join(row_values)}")

    
    table_prompt = "\n".join(lines)
    return table_prompt


def df_to_table_prompt_rowX(df: pd.DataFrame, table_name: str = "dataframe") -> str:
    
    
    lines = [f"-- Table: {table_name}"]
    lines.append("-- Columns:")
    
    
    for col in df.columns:
        col_type = guess_sql_type(df[col].dtypes)
        lines.append(f"--   {col} ({col_type})")

    
    table_prompt = "\n".join(lines)
    return table_prompt
