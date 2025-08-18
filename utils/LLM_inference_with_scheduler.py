import json
import pandas as pd

#from langchain_openai import ChatOpenAI
#from langchain_core.output_parsers import JsonOutputParser
#from langchain_core.prompts import load_prompt
#from pydantic import BaseModel, Field
#from datasets import Dataset

from utils.check_PoT_error import *
from utils.check_text2sql_error import *
from utils.check_exact_match import *

from utils.create_query_format import *

from utils.CoT_inference import *
from utils.PoT_inference import *
from utils.text2sql_inference import *

from utils.PoT_refine_inference import *
from utils.text2sql_refine_inference import *

from utils.run_pandas_code import *
from utils.run_SQL_code import *

from utils.convert_table_datatype import *
from utils.refine_similarity import *
from utils.adjust_context import *
from utils.check_timeout import *




def LLM_inference_with_scheduler(prompt_PoT, prompt_text2sql, prompt_CoT, prompt_refine_PoT, prompt_refine_text2sql, llm, table, question, PoT_results, text2sql_results, CoT_results, scheduler_result , N = 3, sim_threshold = 0.9):
    markdown_df = table.to_markdown(index=False)
    df_code = dataframe_to_code(table)
    table_converted_datatype = convert_to_numeric(table)
    text2sql_df_with_col = df_to_table_prompt(table_converted_datatype, table_name="dataframe")
    table_for_sql_exec = replace_spaces_with_underscores(table_converted_datatype)
    query_PoT_cache = create_reader_request_PoT(df_code, question)
    query_text2sql_cache = create_reader_request_text2sql(text2sql_df_with_col, question)
    query_CoT_cache = create_reader_request_CoT(markdown_df, question)

    try :
        token_count_result = measure_and_adjust_context(prompt_CoT.format(query = query_CoT_cache))
        if token_count_result['adjusted_context'] > 2048:
            llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
            llm_answer_CoT = check_timeout_CoT(lambda: CoT_inference(prompt_CoT, query_CoT_cache, llm), 300)
            CoT_results['N=0'] = llm_answer_CoT
        else:
            llm_answer_CoT = check_timeout_CoT(lambda: CoT_inference(prompt_CoT, query_CoT_cache, llm), 300)
            CoT_results['N=0'] = llm_answer_CoT
    except Exception as e:
        CoT_results['N=0'] = {"solution" : str(e), "answer" : str(e)}


    if scheduler_result == 'PoT':
        try :
            token_count_result = measure_and_adjust_context(prompt_PoT.format(query = query_PoT_cache))
            if token_count_result['adjusted_context'] > 2048:
                llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                llm_answer_PoT = check_timeout_PoT(lambda: PoT_inference(prompt_PoT, query_PoT_cache, llm), 300)
                PoT_code_execution_result = run_pandas_code(llm_answer_PoT["code"], df_code)
                PoT_results['N=0'] = llm_answer_PoT
                PoT_results['N=0 execution_result'] = str(PoT_code_execution_result)    
            else:
                llm_answer_PoT = check_timeout_PoT(lambda: PoT_inference(prompt_PoT, query_PoT_cache, llm), 300)
                PoT_code_execution_result = run_pandas_code(llm_answer_PoT["code"], df_code)
                PoT_results['N=0'] = llm_answer_PoT
                PoT_results['N=0 execution_result'] = str(PoT_code_execution_result)  
        except Exception as e:
            PoT_results['N=0'] = {"code" : str(e)}
            PoT_results['N=0 execution_result'] = str(e)

        for i in range(N):
            try :
                token_count_result = measure_and_adjust_context(prompt_refine_PoT.format(query = query_PoT_cache, code = PoT_results[f"N={i}"]["code"], execution_result = PoT_results[f'N={i} execution_result']))
                if token_count_result['adjusted_context'] > 2048:
                    llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                    llm_refine_answer_PoT = check_timeout_PoT_refine(lambda: refined_PoT_inference(prompt_refine_PoT, query_PoT_cache, PoT_results[f"N={i}"]["code"], PoT_results[f'N={i} execution_result'], llm), 300)
                    refined_PoT_code_execution_result = run_pandas_code(llm_refine_answer_PoT["code"], df_code)
                    PoT_results[f'N={i+1}'] = llm_refine_answer_PoT
                    PoT_results[f'N={i+1} execution_result'] = str(refined_PoT_code_execution_result)
                    PoT_answer = PoT_results[f'N={i+1} execution_result']
                    code_sim = calculate_code_similarity(PoT_results[f'N={i}']['code'], PoT_results[f'N={i+1}']['code'])
                    code_avg_scores = (code_sim['Levenshtein Similarity'] + code_sim['Difflib Similarity'] + code_sim['AST Similarity'] + code_sim['Opcode Similarity']) / 4
                    if (PoT_results[f'N={i} execution_result'] == PoT_results[f'N={i+1} execution_result']) and (code_avg_scores > sim_threshold):
                        break
                else:
                    llm_refine_answer_PoT = check_timeout_PoT_refine(lambda: refined_PoT_inference(prompt_refine_PoT, query_PoT_cache, PoT_results[f"N={i}"]["code"], PoT_results[f'N={i} execution_result'], llm), 300)
                    refined_PoT_code_execution_result = run_pandas_code(llm_refine_answer_PoT["code"], df_code)
                    PoT_results[f'N={i+1}'] = llm_refine_answer_PoT
                    PoT_results[f'N={i+1} execution_result'] = str(refined_PoT_code_execution_result)
                    PoT_answer = PoT_results[f'N={i+1} execution_result']
                    code_sim = calculate_code_similarity(PoT_results[f'N={i}']['code'], PoT_results[f'N={i+1}']['code'])
                    code_avg_scores = (code_sim['Levenshtein Similarity'] + code_sim['Difflib Similarity'] + code_sim['AST Similarity'] + code_sim['Opcode Similarity']) / 4
                    if (PoT_results[f'N={i} execution_result'] == PoT_results[f'N={i+1} execution_result']) and (code_avg_scores > sim_threshold):
                        break
            except Exception as e:
                PoT_results[f'N={i+1}'] = str(e)
                PoT_results[f'N={i+1} execution_result'] = str(e)
                PoT_answer = PoT_results[f'N={i+1} execution_result']
                break
        try:
            if check_PoT_error(PoT_answer) != False and check_exact_match(PoT_answer.replace("[", "").replace("]", "").strip("'").strip('"'), str(CoT_results['N=0']["answer"]).strip("'").strip('"')) == True:
                text2sql_results =  {
                        "N=0": {
                            "code": "<NOTHING>"
                        },
                        "N=0 execution_result": "<NOTHING>",
                        "N=1": {
                            "code": "<NOTHING>"
                        },
                        "N=1 execution_result": "<NOTHING>"
                }
                print("Text2SQL is stopped.")
            else:
                try :
                    token_count_result = measure_and_adjust_context(prompt_text2sql.format(query = query_text2sql_cache))
                    if token_count_result['adjusted_context'] > 2048:
                        llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                        llm_answer_text2sql = check_timeout_text2sql(lambda: text2sql_inference(prompt_text2sql, query_text2sql_cache, llm), 300)
                        text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_answer_text2sql["code"])
                        text2sql_results['N=0'] = llm_answer_text2sql
                        text2sql_results['N=0 execution_result'] = str(text2sql_code_execution_result) 
                    else:
                        llm_answer_text2sql = check_timeout_text2sql(lambda: text2sql_inference(prompt_text2sql, query_text2sql_cache, llm), 300)
                        text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_answer_text2sql["code"])
                        text2sql_results['N=0'] = llm_answer_text2sql
                        text2sql_results['N=0 execution_result'] = str(text2sql_code_execution_result) 
                except Exception as e:
                    text2sql_results['N=0'] = {"code" : str(e)}
                    text2sql_results['N=0 execution_result'] = str(e)
        
                for i in range(N):
                    if ("Error" not in text2sql_results[f'N={i} execution_result']) and (text2sql_results[f'N={i} execution_result'] != "[]") :
                        break
                    try :
                        token_count_result = measure_and_adjust_context(prompt_refine_text2sql.format(query = query_text2sql_cache, code = text2sql_results[f"N={i}"]["code"], execution_result = text2sql_results[f'N={i} execution_result']))
                        if token_count_result['adjusted_context'] > 2048:
                            llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                            llm_refine_answer_text2sql = check_timeout_text2sql_refine(lambda: refined_text2sql_inference(prompt_refine_text2sql, query_text2sql_cache, text2sql_results[f"N={i}"]["code"], text2sql_results[f'N={i} execution_result'], llm), 300)
                            refined_text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_refine_answer_text2sql["code"])
                            text2sql_results[f'N={i+1}'] = llm_refine_answer_text2sql
                            text2sql_results[f'N={i+1} execution_result'] = str(refined_text2sql_code_execution_result)
                            if ("Error" not in text2sql_results[f'N={i+1} execution_result']) and (text2sql_results[f'N={i+1} execution_result'] != "[]") :
                                break
                            if  (text2sql_results[f'N={i}']['code'] == text2sql_results[f'N={i+1}']['code']) and (text2sql_results[f'N={i} execution_result'] == text2sql_results[f'N={i+1} execution_result']):
                                break                
                        else:
                            llm_refine_answer_text2sql = check_timeout_text2sql_refine(lambda: refined_text2sql_inference(prompt_refine_text2sql, query_text2sql_cache, text2sql_results[f"N={i}"]["code"], text2sql_results[f'N={i} execution_result'], llm), 300)
                            refined_text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_refine_answer_text2sql["code"])
                            text2sql_results[f'N={i+1}'] = llm_refine_answer_text2sql
                            text2sql_results[f'N={i+1} execution_result'] = str(refined_text2sql_code_execution_result)
                            if ("Error" not in text2sql_results[f'N={i+1} execution_result']) and (text2sql_results[f'N={i+1} execution_result'] != "[]") :
                                break
                            if  (text2sql_results[f'N={i}']['code'] == text2sql_results[f'N={i+1}']['code']) and (text2sql_results[f'N={i} execution_result'] == text2sql_results[f'N={i+1} execution_result']):
                                break
                    except Exception as e:
                        text2sql_results[f'N={i+1}'] = str(e)
                        text2sql_results[f'N={i+1} execution_result'] = str(e)
                        break
        except Exception as e:
            print("JSON parser Error => text2sql inference")
            try :
                token_count_result = measure_and_adjust_context(prompt_text2sql.format(query = query_text2sql_cache))
                if token_count_result['adjusted_context'] > 2048:
                    llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                    llm_answer_text2sql = check_timeout_text2sql(lambda: text2sql_inference(prompt_text2sql, query_text2sql_cache, llm), 300)
                    text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_answer_text2sql["code"])
                    text2sql_results['N=0'] = llm_answer_text2sql
                    text2sql_results['N=0 execution_result'] = str(text2sql_code_execution_result) 
                else:
                    llm_answer_text2sql = check_timeout_text2sql(lambda: text2sql_inference(prompt_text2sql, query_text2sql_cache, llm), 300)
                    text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_answer_text2sql["code"])
                    text2sql_results['N=0'] = llm_answer_text2sql
                    text2sql_results['N=0 execution_result'] = str(text2sql_code_execution_result) 
            except Exception as e:
                text2sql_results['N=0'] = {"code" : str(e)}
                text2sql_results['N=0 execution_result'] = str(e)
    
            for i in range(N):
                if ("Error" not in text2sql_results[f'N={i} execution_result']) and (text2sql_results[f'N={i} execution_result'] != "[]") :
                    break
                try :
                    token_count_result = measure_and_adjust_context(prompt_refine_text2sql.format(query = query_text2sql_cache, code = text2sql_results[f"N={i}"]["code"], execution_result = text2sql_results[f'N={i} execution_result']))
                    if token_count_result['adjusted_context'] > 2048:
                        llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                        llm_refine_answer_text2sql = check_timeout_text2sql_refine(lambda: refined_text2sql_inference(prompt_refine_text2sql, query_text2sql_cache, text2sql_results[f"N={i}"]["code"], text2sql_results[f'N={i} execution_result'], llm), 300)
                        refined_text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_refine_answer_text2sql["code"])
                        text2sql_results[f'N={i+1}'] = llm_refine_answer_text2sql
                        text2sql_results[f'N={i+1} execution_result'] = str(refined_text2sql_code_execution_result)
                        if ("Error" not in text2sql_results[f'N={i+1} execution_result']) and (text2sql_results[f'N={i+1} execution_result'] != "[]") :
                            break
                        if  (text2sql_results[f'N={i}']['code'] == text2sql_results[f'N={i+1}']['code']) and (text2sql_results[f'N={i} execution_result'] == text2sql_results[f'N={i+1} execution_result']):
                            break                
                    else:
                        llm_refine_answer_text2sql = check_timeout_text2sql_refine(lambda: refined_text2sql_inference(prompt_refine_text2sql, query_text2sql_cache, text2sql_results[f"N={i}"]["code"], text2sql_results[f'N={i} execution_result'], llm), 300)
                        refined_text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_refine_answer_text2sql["code"])
                        text2sql_results[f'N={i+1}'] = llm_refine_answer_text2sql
                        text2sql_results[f'N={i+1} execution_result'] = str(refined_text2sql_code_execution_result)
                        if ("Error" not in text2sql_results[f'N={i+1} execution_result']) and (text2sql_results[f'N={i+1} execution_result'] != "[]") :
                            break
                        if  (text2sql_results[f'N={i}']['code'] == text2sql_results[f'N={i+1}']['code']) and (text2sql_results[f'N={i} execution_result'] == text2sql_results[f'N={i+1} execution_result']):
                            break
                except Exception as e:
                    text2sql_results[f'N={i+1}'] = str(e)
                    text2sql_results[f'N={i+1} execution_result'] = str(e)
                    break

    elif scheduler_result == 'text2sql':
        try :
            token_count_result = measure_and_adjust_context(prompt_text2sql.format(query = query_text2sql_cache))
            if token_count_result['adjusted_context'] > 2048:
                llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                llm_answer_text2sql = check_timeout_text2sql(lambda: text2sql_inference(prompt_text2sql, query_text2sql_cache, llm), 300)
                text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_answer_text2sql["code"])
                text2sql_results['N=0'] = llm_answer_text2sql
                text2sql_results['N=0 execution_result'] = str(text2sql_code_execution_result) 
                text2sql_answer = text2sql_results['N=0 execution_result']
            else:
                llm_answer_text2sql = check_timeout_text2sql(lambda: text2sql_inference(prompt_text2sql, query_text2sql_cache, llm), 300)
                text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_answer_text2sql["code"])
                text2sql_results['N=0'] = llm_answer_text2sql
                text2sql_results['N=0 execution_result'] = str(text2sql_code_execution_result) 
                text2sql_answer = text2sql_results['N=0 execution_result']
        except Exception as e:
            text2sql_results['N=0'] = {"code" : str(e)}
            text2sql_results['N=0 execution_result'] = str(e)
            text2sql_answer = text2sql_results['N=0 execution_result']

        for i in range(N):
            if ("Error" not in text2sql_results[f'N={i} execution_result']) and (text2sql_results[f'N={i} execution_result'] != "[]") :
                break
            try :
                token_count_result = measure_and_adjust_context(prompt_refine_text2sql.format(query = query_text2sql_cache, code = text2sql_results[f"N={i}"]["code"], execution_result = text2sql_results[f'N={i} execution_result']))
                if token_count_result['adjusted_context'] > 2048:
                    llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                    llm_refine_answer_text2sql = check_timeout_text2sql_refine(lambda: refined_text2sql_inference(prompt_refine_text2sql, query_text2sql_cache, text2sql_results[f"N={i}"]["code"], text2sql_results[f'N={i} execution_result'], llm), 300)
                    refined_text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_refine_answer_text2sql["code"])
                    text2sql_results[f'N={i+1}'] = llm_refine_answer_text2sql
                    text2sql_results[f'N={i+1} execution_result'] = str(refined_text2sql_code_execution_result)
                    text2sql_answer = text2sql_results[f'N={i+1} execution_result']
                    if ("Error" not in text2sql_results[f'N={i+1} execution_result']) and (text2sql_results[f'N={i+1} execution_result'] != "[]") :
                        break
                    if  (text2sql_results[f'N={i}']['code'] == text2sql_results[f'N={i+1}']['code']) and (text2sql_results[f'N={i} execution_result'] == text2sql_results[f'N={i+1} execution_result']):
                        break                
                else:
                    llm_refine_answer_text2sql = check_timeout_text2sql_refine(lambda: refined_text2sql_inference(prompt_refine_text2sql, query_text2sql_cache, text2sql_results[f"N={i}"]["code"], text2sql_results[f'N={i} execution_result'], llm), 300)
                    refined_text2sql_code_execution_result = execute_single_query(table_for_sql_exec, llm_refine_answer_text2sql["code"])
                    text2sql_results[f'N={i+1}'] = llm_refine_answer_text2sql
                    text2sql_results[f'N={i+1} execution_result'] = str(refined_text2sql_code_execution_result)
                    text2sql_answer = text2sql_results[f'N={i+1} execution_result']
                    if ("Error" not in text2sql_results[f'N={i+1} execution_result']) and (text2sql_results[f'N={i+1} execution_result'] != "[]") :
                        break
                    if  (text2sql_results[f'N={i}']['code'] == text2sql_results[f'N={i+1}']['code']) and (text2sql_results[f'N={i} execution_result'] == text2sql_results[f'N={i+1} execution_result']):
                        break
            except Exception as e:
                text2sql_results[f'N={i+1}'] = str(e)
                text2sql_results[f'N={i+1} execution_result'] = str(e)
                text2sql_answer = text2sql_results[f'N={i+1} execution_result']
                break
        try:
            if check_text2sql_error(text2sql_answer) != False and check_exact_match(text2sql_answer.replace("[", "").replace("]", "").strip("'").strip('"'), str(CoT_results['N=0']["answer"]).strip("'").strip('"')) == True:
                PoT_results =  {
                        "N=0": {
                            "code": "<NOTHING>"
                        },
                        "N=0 execution_result": "<NOTHING>",
                        "N=1": {
                            "code": "<NOTHING>"
                        },
                        "N=1 execution_result": "<NOTHING>"
                }
                print("PoT is stopped.")
            else:
                print("JSON parser Error => PoT inference")
                try :
                    token_count_result = measure_and_adjust_context(prompt_PoT.format(query = query_PoT_cache))
                    if token_count_result['adjusted_context'] > 2048:
                        llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                        llm_answer_PoT = check_timeout_PoT(lambda: PoT_inference(prompt_PoT, query_PoT_cache, llm), 300)
                        PoT_code_execution_result = run_pandas_code(llm_answer_PoT["code"], df_code)
                        PoT_results['N=0'] = llm_answer_PoT
                        PoT_results['N=0 execution_result'] = str(PoT_code_execution_result)    
                    else:
                        llm_answer_PoT = check_timeout_PoT(lambda: PoT_inference(prompt_PoT, query_PoT_cache, llm), 300)
                        PoT_code_execution_result = run_pandas_code(llm_answer_PoT["code"], df_code)
                        PoT_results['N=0'] = llm_answer_PoT
                        PoT_results['N=0 execution_result'] = str(PoT_code_execution_result)  
                except Exception as e:
                    PoT_results['N=0'] = {"code" : str(e)}
                    PoT_results['N=0 execution_result'] = str(e)
    
                for i in range(N):
                    try :
                        token_count_result = measure_and_adjust_context(prompt_refine_PoT.format(query = query_PoT_cache, code = PoT_results[f"N={i}"]["code"], execution_result = PoT_results[f'N={i} execution_result']))
                        if token_count_result['adjusted_context'] > 2048:
                            llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                            llm_refine_answer_PoT = check_timeout_PoT_refine(lambda: refined_PoT_inference(prompt_refine_PoT, query_PoT_cache, PoT_results[f"N={i}"]["code"], PoT_results[f'N={i} execution_result'], llm), 300)
                            refined_PoT_code_execution_result = run_pandas_code(llm_refine_answer_PoT["code"], df_code)
                            PoT_results[f'N={i+1}'] = llm_refine_answer_PoT
                            PoT_results[f'N={i+1} execution_result'] = str(refined_PoT_code_execution_result)
                            code_sim = calculate_code_similarity(PoT_results[f'N={i}']['code'], PoT_results[f'N={i+1}']['code'])
                            code_avg_scores = (code_sim['Levenshtein Similarity'] + code_sim['Difflib Similarity'] + code_sim['AST Similarity'] + code_sim['Opcode Similarity']) / 4
                            if (PoT_results[f'N={i} execution_result'] == PoT_results[f'N={i+1} execution_result']) and (code_avg_scores > sim_threshold):
                                break
                        else:
                            llm_refine_answer_PoT = check_timeout_PoT_refine(lambda: refined_PoT_inference(prompt_refine_PoT, query_PoT_cache, PoT_results[f"N={i}"]["code"], PoT_results[f'N={i} execution_result'], llm), 300)
                            refined_PoT_code_execution_result = run_pandas_code(llm_refine_answer_PoT["code"], df_code)
                            PoT_results[f'N={i+1}'] = llm_refine_answer_PoT
                            PoT_results[f'N={i+1} execution_result'] = str(refined_PoT_code_execution_result)
                            code_sim = calculate_code_similarity(PoT_results[f'N={i}']['code'], PoT_results[f'N={i+1}']['code'])
                            code_avg_scores = (code_sim['Levenshtein Similarity'] + code_sim['Difflib Similarity'] + code_sim['AST Similarity'] + code_sim['Opcode Similarity']) / 4
                            if (PoT_results[f'N={i} execution_result'] == PoT_results[f'N={i+1} execution_result']) and (code_avg_scores > sim_threshold):
                                break
                    except Exception as e:
                        PoT_results[f'N={i+1}'] = str(e)
                        PoT_results[f'N={i+1} execution_result'] = str(e)
                        break
        except Exception as e:
            try :
                token_count_result = measure_and_adjust_context(prompt_PoT.format(query = query_PoT_cache))
                if token_count_result['adjusted_context'] > 2048:
                    llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                    llm_answer_PoT = check_timeout_PoT(lambda: PoT_inference(prompt_PoT, query_PoT_cache, llm), 300)
                    PoT_code_execution_result = run_pandas_code(llm_answer_PoT["code"], df_code)
                    PoT_results['N=0'] = llm_answer_PoT
                    PoT_results['N=0 execution_result'] = str(PoT_code_execution_result)    
                else:
                    llm_answer_PoT = check_timeout_PoT(lambda: PoT_inference(prompt_PoT, query_PoT_cache, llm), 300)
                    PoT_code_execution_result = run_pandas_code(llm_answer_PoT["code"], df_code)
                    PoT_results['N=0'] = llm_answer_PoT
                    PoT_results['N=0 execution_result'] = str(PoT_code_execution_result)  
            except Exception as e:
                PoT_results['N=0'] = {"code" : str(e)}
                PoT_results['N=0 execution_result'] = str(e)
            for i in range(N):
                try :
                    token_count_result = measure_and_adjust_context(prompt_refine_PoT.format(query = query_PoT_cache, code = PoT_results[f"N={i}"]["code"], execution_result = PoT_results[f'N={i} execution_result']))
                    if token_count_result['adjusted_context'] > 2048:
                        llm = llm_adjusted_context(llm, token_count_result['adjusted_context'])
                        llm_refine_answer_PoT = check_timeout_PoT_refine(lambda: refined_PoT_inference(prompt_refine_PoT, query_PoT_cache, PoT_results[f"N={i}"]["code"], PoT_results[f'N={i} execution_result'], llm), 300)
                        refined_PoT_code_execution_result = run_pandas_code(llm_refine_answer_PoT["code"], df_code)
                        PoT_results[f'N={i+1}'] = llm_refine_answer_PoT
                        PoT_results[f'N={i+1} execution_result'] = str(refined_PoT_code_execution_result)
                        code_sim = calculate_code_similarity(PoT_results[f'N={i}']['code'], PoT_results[f'N={i+1}']['code'])
                        code_avg_scores = (code_sim['Levenshtein Similarity'] + code_sim['Difflib Similarity'] + code_sim['AST Similarity'] + code_sim['Opcode Similarity']) / 4
                        if (PoT_results[f'N={i} execution_result'] == PoT_results[f'N={i+1} execution_result']) and (code_avg_scores > sim_threshold):
                            break
                    else:
                        llm_refine_answer_PoT = check_timeout_PoT_refine(lambda: refined_PoT_inference(prompt_refine_PoT, query_PoT_cache, PoT_results[f"N={i}"]["code"], PoT_results[f'N={i} execution_result'], llm), 300)
                        refined_PoT_code_execution_result = run_pandas_code(llm_refine_answer_PoT["code"], df_code)
                        PoT_results[f'N={i+1}'] = llm_refine_answer_PoT
                        PoT_results[f'N={i+1} execution_result'] = str(refined_PoT_code_execution_result)
                        code_sim = calculate_code_similarity(PoT_results[f'N={i}']['code'], PoT_results[f'N={i+1}']['code'])
                        code_avg_scores = (code_sim['Levenshtein Similarity'] + code_sim['Difflib Similarity'] + code_sim['AST Similarity'] + code_sim['Opcode Similarity']) / 4
                        if (PoT_results[f'N={i} execution_result'] == PoT_results[f'N={i+1} execution_result']) and (code_avg_scores > sim_threshold):
                            break
                except Exception as e:
                    PoT_results[f'N={i+1}'] = str(e)
                    PoT_results[f'N={i+1} execution_result'] = str(e)
                    break
    return PoT_results, text2sql_results, CoT_results



