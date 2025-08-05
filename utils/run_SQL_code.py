import pandas as pd
import sqlite3



connection = sqlite3.connect(":memory:")  


def execute_single_query(df, query, table_name: str = "dataframe"):
    df.to_sql(table_name, connection, index=False, if_exists="replace")  
    try:
        
        result = pd.read_sql_query(query, connection)
        
        return result.values.tolist()
    except Exception as e:
        
        return f"Error: {e}"


def replace_spaces_with_underscores(df):
    """
    Replace spaces in column names of a DataFrame with underscores.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with updated column names.
    """
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.str.replace(' ', '_', regex=False)
    return df_copy