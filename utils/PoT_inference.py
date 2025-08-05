import json
from langchain_ollama import ChatOllama
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser



def PoT_inference(prompt, query, llm):
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    try:
        llm_answer = chain.invoke({"query": query})
        return llm_answer
    except Exception as e:
        try:
            new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            chain = prompt | llm | new_parser
            llm_answer = chain.invoke({"query": query})
            return llm_answer
        except Exception as e2:
            return {"code" : str(e2)}
