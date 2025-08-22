# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List
from .table_custom_linearize import TableLinearize
from .table_custom_truncate import TableTruncate

with open("prompts/text2sql_prompt.txt", "r") as f:
    text2sql_prompt = f.read()

class TableProcessor(object):

    def __init__(self, table_linearize_func: TableLinearize,
                 table_truncate_funcs: List[TableTruncate],
                 target_delimiter: str = ", ",
                 tokenizer=None):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs
        self.target_delimiter = target_delimiter
        self.tokenizer = tokenizer

    def process_input(self, table_content: Dict, question: str, answer: List[str], tableqa: bool, fewshot: bool, tabfact: bool) -> str:
        """
        Preprocess a sentence into the expected format for model translate.
        """
        # modify a table internally
        for truncate_func in self.table_truncate_funcs:
            truncate_func.truncate_table(table_content, question, answer)
        # linearize a table into a string
        linear_table = self.table_linearize_func.process_table(table_content)
        # concat question with linear_table
        input = ""
        if not tableqa:
            if fewshot:
                input += text2sql_prompt + "\n"
            input += "Read the following table and then write SQL code to answer the question:\n"
            input += linear_table + "\n"
            if tabfact:
                input += f"Question: Determine whether the statement is True or False. (Return 'yes' or 'no')\n{question}\n"
            else:
                input += f"Question: {question}\n"
            input += "\n## Return a query for the 'SQL code' with one key: code. Respond using JSON only."
            
        else: 
            input = question + "\n" + linear_table
            if tabfact: 
                input += '''\nLet's think step by step to verify the statement, and then give the final answer. Ensure the final answer format is only "Final Answer: yes/no" form, no other form. And ensure the final answer is only a Yes or No, without any explanation'''
            else:
                input += '''\nLet's think step by step, and then give the final answer. Ensure the final answer format is only "Final Answer: Answer" form, no other form. And ensure the final answer is a number or entity names, as short as possible, without any explanation.'''
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input}],
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def process_output(self, answer: List[str]) -> str:
        """
        Flatten the output for translation
        """
        output = self.target_delimiter.join(answer)
        if output.strip() == "":
            raise Exception("The Answer is EMPTY!")
        else:
            return output
