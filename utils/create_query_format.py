import pandas as pd

def create_reader_request_CoT(df, question):
    string = f'Read the following table and then write solution texts to answer a question:\n\n'
    string += df + '\n\n'
    string += f'Question: {question}'
    string += '\n## Return a query for the solution and answer with two keys: solution and answer. Respond using JSON only.\n\n'
    return string


def dataframe_to_code(df):
    # Extract column names and their values
    data = {col: df[col].tolist() for col in df.columns}

    # Convert the DataFrame index to a list
    index = df.index.tolist()

    # Create the Python code
    code = "data = {\n"
    for col, values in data.items():
        code += f"    '{col}': {values},\n"
    code += "}\n"
    
    # Add the DataFrame creation code
    code += f"df = pd.DataFrame(data, index={index})"

    return code


def create_reader_request_PoT(df_code, question):
    string = f'Read the following table and then write Python code with pandas to answer a question:\n\n'
    string += 'import pandas as pd\n' + df_code + '\n\n'
    string += f'Question: {question}'
    string += "\n## You don’t need to reprint pre-written code like `import pandas as pd`, `data = {{...}}`, or `df = pd.DataFrame(data)`. That code will be provided separately, so just give me the code that processes `data` and `df`."
    string += "\n## Return a query for the python code with pandas which return ans with one key: code. Respond using JSON only.\n\n"
    return string

def create_reader_request_text2sql(text2sql_raw_df, question):
    string = f'Read the following table and then write SQL code to answer the question:\n\n'
    string += text2sql_raw_df + '\n\n'
    string += f'Question: {question}'
    string += "\n## Return a query for the 'SQL code' with one key: code. Respond using JSON only.\n\n"
    return string



# for tabfact

def create_reader_request_CoT_for_tabfact(df, question, caption):
    string = f'Read the following table and then write solution texts to answer a question:\n\n'
    string += df + '\n\n'
    string += f'Question: It is about {caption}. Determine whether the statement is True or False : {question}'
    string += '\n## Return a query for the solution and answer with two keys: solution and answer. Respond using JSON only.\n\n'
    return string



def create_reader_request_PoT_for_tabfact(df_code, question, caption):
    string = f'Read the following table and then write Python code with pandas to answer a question:\n\n'
    string += 'import pandas as pd\n' + df_code + '\n\n'
    string += f'Question: It is about {caption}. Determine whether the statement is True or False : {question}'
    string += "\n## You don’t need to reprint pre-written code like `import pandas as pd`, `data = {{...}}`, or `df = pd.DataFrame(data)`. That code will be provided separately, so just give me the code that processes `data` and `df`."
    string += "\n## Return a query for the python code with pandas which return ans with one key: code. Respond using JSON only.\n\n"
    return string

def create_reader_request_text2sql_for_tabfact(text2sql_raw_df, question, caption):
    string = f'Read the following table and then write SQL code to answer the question:\n\n'
    string += text2sql_raw_df + '\n\n'
    string += f'Question: It is about {caption}. Determine whether the statement is True or False : {question}'
    string += "\n## Return a query for the 'SQL code' with one key: code. Respond using JSON only.\n\n"
    return string
