# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Utils for linearizing the table content into a flatten sequence
"""
import abc
import abc
from typing import Dict, List, List
import pandas as pd


class TableLinearize(abc.ABC):

    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


class IndexedRowTableLinearize(TableLinearize):

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        # df = pd.DataFrame(table_content['rows'], columns=table_content['header'])
        # return df_to_table_prompt(df)

        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE

        headers = self.process_header(table_content["header"])
        rows = [self.process_row(row, i) for i, row in enumerate(table_content["rows"])]
        table_str  = headers
        table_str += "\n--\n"
        table_str += "-- Rows:\n"
        table_str += "\n".join(rows)
        return table_str

    
    def process_header(self, headers: List):
        lines = []
        lines.append('-- Columns:')
        for col in headers:
            lines.append(f'--   {col}')
        return '\n'.join(lines)
    
    def process_row(self, row: List, row_index: int):
        row = [str(val) for val in row]
        return f"--   {' | '.join(row)}"

def guess_sql_type(dtype) -> str:
    """
    Pandas dtype을 보고 SQL에서 자주 쓰이는 자료형으로 매핑해주는 함수 예시입니다.
    상황과 DBMS에 따라 원하는 자료형을 추가/조정하세요.
    """
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "DECIMAL"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    else:
        return "TEXT"

def df_to_table_prompt(df: pd.DataFrame) -> str:
    lines = []
    lines.append("-- Columns:")
    
    for col in df.columns:
        col_type = guess_sql_type(df[col].dtypes)
        # lines.append(f"--   {col} ({col_type})")
        lines.append(f"--   {col}")

    lines.append("--")
    lines.append("-- Rows:")
    for idx, row in df.iterrows():
        row_values = [str(val) for val in row.values]
        lines.append(f"--   {' | '.join(row_values)}")

    table_prompt = "\n".join(lines)
    return table_prompt