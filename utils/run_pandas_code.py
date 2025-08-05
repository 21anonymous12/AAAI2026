import pandas as pd

def run_code(code: str):
    
    local_vars = {}
    
    try:
        exec(code, {}, local_vars)
        return local_vars['ans']  
    except Exception as e:
        
        return str(e)


def generate_pandas_code(original_code: str, df_code: str) -> str:
    new_code = "import pandas as pd\n\nimport numpy as np\n\n"


    new_code += df_code.strip() + "\n\n"


    new_code += original_code.strip() + "\n"

    return new_code

def run_pandas_code(original_code: str, df_code: str):
    
    final_code = generate_pandas_code(original_code, df_code)
    

    result = run_code(final_code.replace("'''", ""))
    return result