import re

def extract_answer(text:str, patterns:list = [r"Final Answer:(.*)", r": (.*)", r"is (.*)"], return_match_flag=False):
    """
    Extracts the answer from a response text.

    Parameters:
    text (str): The response text.

    Returns:
    str: The extracted answer.
    """
    # Regular expression patterns
    patterns = patterns
    answer = None
    match_flag = False

    # convert text to lower case to ignore case
    text = text.lower()

    for pattern in patterns:
        # find matches
        matches = re.findall(pattern, text, re.IGNORECASE)
        # if matches found, update answer with the last match
        if matches:
            answer = matches[-1]
            if "final answer" in pattern.lower():
                match_flag = True
            
            if return_match_flag:
                return answer, match_flag
            return answer.replace('final answer:', '').strip()

    if return_match_flag:
        return answer, match_flag
    return answer