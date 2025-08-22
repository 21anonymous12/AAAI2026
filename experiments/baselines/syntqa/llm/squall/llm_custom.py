from openai import OpenAI
import pandas as pd
import string
import re
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM, SamplingParams


file_path = "predict/tablebench_classifier_test10.csv"
provider = 'openai' # huggingface, vllm, openai

MODEL_NAME = "meta-llama/llama-3.2-3b-instruct"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# MODEL_NAME = "google/gemma-2-2b-it"

if provider == 'openai':
    client = OpenAI()
elif provider == 'vllm':
    model = LLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        tokenizer_mode="auto",
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
else:
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

df = pd.read_csv(file_path)

df['truncated_tableqa'] = pd.to_numeric(df['truncated_tableqa'])
df = df.reset_index(drop=True)
df.loc[:,'gpt_score'] = 0
df.loc[:, 'pred'] = 'none'
print('df size: ', df.shape)

##############

with open('llm/squall/relevance-prompt.txt', 'r') as f:
    entity_align_prompt = f.read()

with open('llm/squall/alignment-prompt.txt', 'r') as f:
    number_align_prompt = f.read()  

with open('llm/squall/similarity-prompt.txt', 'r') as f:
    similar_prompt = f.read()   

with open('llm/squall/comparison-prompt.txt', 'r') as f:
    compare_prompt = f.read()   

with open('llm/squall/contradiction-prompt.txt', 'r') as f:
    contradiction_prompt = f.read()   

##############
def checkDigit(res):
    res = str(res)
    return res.replace('.','').replace(',','').isdigit()

def call_model(cur_prompt, stop, temperature = 0, system_prompt = None):
    if provider == 'openai':
        
        ans = client.chat.completions.create(
                    model='gpt-4o',
                    messages = [
                        {"role": "user", "content": cur_prompt}
                    ],
                    temperature=temperature)
        returned = ans.choices[0].message.content
        
        return returned

    elif provider == 'huggingface':
        
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": cur_prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=temperature,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    elif provider == 'vllm':
        sampling_params = SamplingParams(
            temperature=temperature,
            stop=stop
        )
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": cur_prompt})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = model.generate([text], sampling_params=sampling_params)
        response = outputs[0].outputs[0].text
        return response
    else:
        raise NotImplementedError(f"Provider {provider} is not supported.")
    

def similarAlign(question, response1, response2):

    prompt = similar_prompt + "\n\nQuestion: " + question + '\nResponse A: ' + response1 + '\nResponse B: ' + response2 + '\nAnswer: '
    print('\n\n///similarity///')
    print(prompt)
    gen = call_model(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_model(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)
    gen = gen.lower().strip()
    print('Answer: ', gen)
    if gen == 'yes':
        return True
    else:
        return False
    
def entityAlign(question, response):

    prompt = entity_align_prompt + "\n\nQuestion: " + question + '\nResponse: ' + response + '\nAnswer: '
    print('\n\n///entityAlign///')
    print(prompt)
    gen = call_model(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_model(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)
    gen = gen.lower().strip()
    print('Answer: ', gen)
    if gen == 'yes':
        return True
    else:
        return False

def numberAlign(question, response):

    prompt = number_align_prompt + "\n\nQuestion: " + question + '\nResponse: ' + response + '\nAnswer: '
    print('\n\n///numberAlign///')
    print(prompt)
    gen = call_model(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_model(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)
    gen = gen.lower().strip()
    print('Answer: ', gen)
    if gen == 'yes':
        return True
    else:
        return False
    

def countNumber(table, question):

    prompt = contradiction_prompt + "\n\nTable: " + table + '\nQuestion: ' + question + '\nAnswer: '
    print('\n\n///contradiction///')
    print(prompt)

    def get_number(str):
        numbers = re.findall(r'\d+', str)
        numbers = [int(num) for num in numbers]
        return numbers

    gen = call_model(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2):
        if len(get_number(gen))>0:
            break
        gen = call_model(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)

    number = get_number(gen)
    number = number[-1] if len(number)>0 else 0
    print('Answer: ', number)
    return number

def main():
    for i, row in df.iterrows():

        if i < 0 and row['gpt_score'] in [1, 0]:
            continue

        print('\n----row: ', i, '-----')
        # if i > 704:
            # break

        question = row['question']
        question = question.replace(' -lrb- ','(').replace(' -rrb-',')')
        ans_tableqa = str(row['ans_tableqa']).lower().strip()
        ans_text_to_sql = str(row['ans_text_to_sql']).lower().strip()
        acc_tableqa = int(row['acc_tableqa'])
        acc_text_to_sql = int(row['acc_text_to_sql'])
        truncated = int(row['truncated_tableqa'])
        table = 'header: ' + row['input_tokens'].split('-- columns:')[1].split("let's")[0]

        nl_response_text_to_sql = []
        for res in ans_text_to_sql.split('|'):
            if res.lower() != 'none' and res not in nl_response_text_to_sql:
                nl_response_text_to_sql.append(res)
        response_text_to_sql = ', '.join(nl_response_text_to_sql)

        nl_response_tableqa = []
        for res in ans_tableqa.split('|'):
            if res not in nl_response_tableqa:
                nl_response_tableqa.append(res)
        response_tableqa = ', '.join(nl_response_tableqa)

        if acc_tableqa==acc_text_to_sql:
            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
            df.loc[i, 'pred'] = df.loc[i, 'ans_text_to_sql']
            print('CASE A: acc_tableqa = acc_text_to_sql')
            continue

        if acc_tableqa!=acc_text_to_sql and ans_text_to_sql in ['', 'nan', 'na', 'none']:
            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
            df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
            print('CASE B: acc_text_to_sql => nan')
            continue

        ##########################################
        # answer format correction 
        ##########################################

        # 6 pts vs 6
        if checkDigit(ans_text_to_sql) or checkDigit(ans_tableqa):
            if len(ans_text_to_sql) > len(ans_tableqa):
                shorter = ans_tableqa
                longer = ans_text_to_sql
                words_longer = longer.split()
                if len(words_longer)==2 and words_longer[0]==shorter and checkDigit(shorter):
                    print('CASE D: ', longer, shorter)
                    df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                    df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
                    print('choose tableqa ', ans_tableqa)
                    continue
            else:
                shorter = ans_text_to_sql
                longer = ans_tableqa
                words_longer = longer.split()
                if len(words_longer)==2 and words_longer[0]==shorter and checkDigit(shorter):
                    print('CASE D: ', longer, shorter)
                    df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                    df.loc[i, 'pred'] = df.loc[i, 'ans_text_to_sql']
                    print('choose text_to_sql ', ans_text_to_sql)
                    continue

        # at denver broncos vs denver broncos
        if response_text_to_sql == 'at ' + response_tableqa:
            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
            df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
            continue

        # 98,453 vs 98453
        if response_text_to_sql.replace(',','') == response_tableqa.replace(',',''):
            if ',' in response_tableqa:
                if response_text_to_sql.isdigit():
                    df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                    df.loc[i, 'pred'] = df.loc[i, 'ans_text_to_sql']
                    continue
            else:
                if response_tableqa.isdigit():
                    df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                    df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
                    continue
        
        # no vs 0
        if response_text_to_sql=='0' and response_tableqa=='no':
            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
            df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
            continue
        if response_text_to_sql=='1' and response_tableqa=='yes':
            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
            df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
            continue

        if acc_tableqa!=acc_text_to_sql:

            ##########################################
            # answer entity and number of answer aligment
            ##########################################

            if checkDigit(row['ans_text_to_sql']) and checkDigit(row['ans_tableqa']):
                df.loc[i, 'entityAlign'] = 1
                df.loc[i, 'numberAlign'] = 1
                df.loc[i, 'similarity'] = 1
            else:
                if nl_response_tableqa[0]==nl_response_text_to_sql[0]:
                    similarity = True
                else:
                    similarity = similarAlign(question, nl_response_text_to_sql[0], nl_response_tableqa[0])
                df.loc[i, 'similarity'] = int(similarity)
                if df.loc[i, 'similarity']==1:
                    df.loc[i, 'entityAlign'] = 1
                    if 'name a ' in question and len(nl_response_text_to_sql)>1:
                        number_align = False
                    elif len(nl_response_text_to_sql)>1:
                        number_align = numberAlign(question, response_text_to_sql)
                    else:
                        number_align = True
                    df.loc[i, 'numberAlign'] = int(number_align)
                else:
                    entity_align = entityAlign(question, response_text_to_sql)
                    df.loc[i, 'entityAlign'] = int(entity_align)

                    if df.loc[i, 'entityAlign']==1:
                        if 'name a ' in question and len(nl_response_text_to_sql)>1:
                            number_align = False
                        elif len(nl_response_text_to_sql)>1:
                            number_align = numberAlign(question, response_text_to_sql)
                        else:
                            number_align = True
                        df.loc[i, 'numberAlign'] = int(number_align)
                    else:
                        df.loc[i, 'numberAlign'] = 0
            
            if df.loc[i, 'numberAlign'] == 1:
                if truncated==1:
                    print('CASE E: pass alignment check, table is truncated. ', response_text_to_sql)

                    ##########################################
                    # contradiction in counting
                    ##########################################

                    try:
                        number_text_to_sql = float(response_text_to_sql)
                    except Exception as e:
                        number_text_to_sql = None
                    if number_text_to_sql is not None and number_text_to_sql.is_integer() and number_text_to_sql<=20:
                        gpt_count = countNumber(table, question)
                        if gpt_count>number_text_to_sql:
                            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                            df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
                            print(f'CASE E-1: counting contradiction. GPT: {gpt_count} > Text-to-SQL: {number_text_to_sql}')
                        else:
                            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                            df.loc[i, 'pred'] = df.loc[i, 'ans_text_to_sql']
                    else:
                        df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                        df.loc[i, 'pred'] = df.loc[i, 'ans_text_to_sql']
                    continue
                else:

                    ##########################################
                    # compare two options
                    ##########################################
                    
                    content = compare_prompt + '\n'
                    content += "\n\nTable: " + table + "\nQuestion: " + question + '\nResponse A: ' + response_text_to_sql+ '\nResponse B: ' + response_tableqa + '\nAnswer: '
                    print('\n\n')
                    print(content)
                    completion = call_model(content, ['\n'], system_prompt = "You are an advanced AI capable of analyzing and understanding information within tables.")
                res = completion
                df.loc[i, 'comparison'] = res

                sents = [
                    'final answer is b',
                    'final answer: b',
                    'final answer:b',
                    'correct answer is b'
                ]
                conditions = any([x in res for x in sents])
                if conditions:
                    df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                    df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
                else:
                    df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                    df.loc[i, 'pred'] = df.loc[i, 'ans_text_to_sql']
                print('CASE F: compare two answers. ', 'B is correct: ', conditions)
                print('GPT response: ', res)
                continue
            else:
                print('CASE G: text-to-sql answer aligment check fails, choose tableqa')
                df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                df.loc[i, 'pred'] = df.loc[i, 'ans_tableqa']
                continue

        raise NotImplementedError

    gpt_scores = df.loc[:,'gpt_score'].values
    gpt_scores = [x for x in gpt_scores if not isinstance(x, float) or not math.isnan(x)]
    oracle = df.loc[:,'oracle'].values[:len(gpt_scores)]

    print('\n\navg oracle: ', np.round(np.mean(oracle),4))
    print('avg gpt: ', np.round(np.mean(gpt_scores),4))

    df.to_csv(file_path, index=False)



if __name__ == '__main__':
    main()