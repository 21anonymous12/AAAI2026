import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import load_prompt
import ast
from typing import List, Dict
import argparse
import yaml
import os
import json
from langchain_ollama import ChatOllama
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

llm_FM = ChatOllama(
    model="qwen2.5:0.5b-instruct-q8_0", 
    format="json",  
    temperature=0,
    num_ctx = 1024,  
    )


def core_extract_inference(prompt, question, answer , llm):
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    try:
        llm_answer = chain.invoke({"Question": question, "Answer": answer})
        return llm_answer
    except Exception as e:
        try:
            new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            chain = prompt | llm | new_parser
            llm_answer = chain.invoke({"Question": question, "Answer": answer})
            return llm_answer
        except Exception as e2:
            return {"Extracted_Answer" : str(e2)}



def FM_inference(prompt, question, LLM_ans):
    if isinstance(LLM_ans, str) and len(LLM_ans) > 100:
        try:
            llm_final_answer = core_extract_inference(prompt, question, LLM_ans, llm_FM)
            return llm_final_answer['Extracted_Answer']
        except KeyError:
            return LLM_ans
    else:
        return LLM_ans