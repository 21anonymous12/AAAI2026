import re
from typing import Callable, Any


error_PoT_pattern = ["Can only use .str accessor with string values!", ".str accessor", "unsupported operand", "Reindexing only valid", "Series is ambiguous", "arg must be", "string values!", "Non-numeric values", "Non-numeric", "<NOTHING>","nameerror","typeerror","<unknown>","keyerror","syntaxerror","indexerror", "valueerror", "dtype='int64'", "no overlapping index names", "cannot reindex", "duplicate labels", "bytes-like object", "is not defined", "argument must be a string or a real number", "'NAType'", "first argument must be an iterable of pandas objects", "you passed an object of type", "Error: ", "cannot be concatenated", "cannot join with no overlapping index names","must be of the same length","does not match format","scalar, tuple, or list","NoneType","must be a string","None of [Index(", "invalid literal for ", "out of bounds for", "has no attribute", "invalid syntax", "(<string>, line", "dtype: object", "not supported between instances", "list index out of range",  "is out-of-bounds", "unhashable type:", "None of [Index", "invalid index", "index out of range", "not convert string to float", "not found in axis", "The truth value of a Series is ambiguous.", "Length mismatch: Expected ", "get argmax of an empty sequence", "Unable to parse string", 'arg is an empty sequence', 'No unique match found', '(expected ', 'not enough values to unpack', 'expected hh:mm:ss format before', ' object is not iterable', "doesn't match format", "dtype: int64", "can only concatenate str", "unsupported operand type(s) for", "pattern contains no capture groups", "nothing to repeat at position", "takes no keyword arguments", "Something has gone wrong, please report a bug at", "No valid entry dates", "attempt to get argmin", "No specific type mentioned", "column not found in data", "arg must be a list, tuple, 1-d array, or Series", "did not contain a loop with signature matching types", "<class ", "cannot convert float", "nomination found for", "No valid attendance data available", "Length of values", "not match length of index", "Invalid comparison between", "Data not available in the provided DataFrame", "unexpected keyword argument", "not supported for the", "No specific issue range found for", "does not support strftime", "No valid", "unconverted data remains when parsing with format", "could not be broadcast", "operands could not", "object do not match", "Unalignable boolean Series", "Boolean array expected", "Series.replace cannot use dict-like to_replace and non-None value", "Expected value of kwarg", "No matching driver combination found", "unit must not be specified", "if the input contains a str", "integer or boolean arrays are valid indices"]



error_PoT_match_pattern = ["<NOTHING>","code", "'code'", "'index'", "'ans'", "nan", "None", "[nan]", "ans", "$nan", "[]"]


def check_PoT_error(text):
    if len(text) > 150:
        return False
    for pattern in error_PoT_pattern:
        if pattern in text:
            return False
    for pattern in error_PoT_match_pattern:
        if text == pattern:
            return False
    return text


error_text2sql_pattern = ["<NOTHING>",'Error:', '[nan]']



error_text2sql_match_pattern = ["<NOTHING>", "code", "'code'", '[]']


def check_text2sql_error(text):
    if len(text) > 150:
        return False
    for pattern in error_text2sql_pattern:
        if pattern in text:
            return False
    for pattern in error_text2sql_match_pattern:
        if text == pattern:
            return False
    return text



def long_text_check(
    text: str,
    tokenizer: Any,
    check_PoT_error: Callable[[str], bool],
    check_text2sql_error: Callable[[str], bool]
) -> str:
    """
    Process `text` as described:
      1) Tokenize without truncation.
      2) If ≤2800 tokens, return original.
      3) Else compute spans between <PoT>…</PoT> and <text2sql>…</text2sql>.
         Let primary be the longer (or text2sql if equal), secondary the other.
      4) On original text, extract all <N=??_execution_result>…</N=??_execution_result>
         in the primary section; run check_primary_error on each.
      5) If all primary checks are False, replace primary inner contents with:
         <N=0_code><NOTHING></N=0_code>
         <N=0_execution_result><NOTHING></N=0_execution_result>
         <N=1_code><NOTHING></N=1_code>
         <N=1_execution_result><NOTHING></N=1_execution_result>
         and return.
      6) Otherwise, extract and check the secondary section similarly.
         If all secondary checks are False, replace secondary inner contents as above and return.
      7) If both primary and secondary each have at least one True from their checks,
         replace the primary section as above and return.
      8) Else return the original text.
    """
    original = text

    # 1 & 2: tokenize and early exit
    token_ids = tokenizer.encode(text, truncation=False)
    if len(token_ids) <= 2800:
        return original

    # 3: compute spans
    def span_between(start_id: int, end_id: int) -> int:
        try:
            s = token_ids.index(start_id)
            e = token_ids.index(end_id, s+1)
            return e - s - 1
        except ValueError:
            return original

    pot_span = span_between(128011, 128028)
    t2s_span = span_between(128029, 128030)

    # choose primary/secondary
    if pot_span > t2s_span:
        primary_tag, primary_check = "PoT", check_PoT_error
        secondary_tag, secondary_check = "text2sql", check_text2sql_error
    else:
        primary_tag, primary_check = "text2sql", check_text2sql_error
        secondary_tag, secondary_check = "PoT", check_PoT_error

    # helpers to extract results and replace a section
    def extract_results(txt: str, tag: str) -> list[str]:
        m = re.search(rf"(?s)<{tag}>(.*?)</{tag}>", txt)
        if not m:
            return []
        inner = m.group(1)
        return [res.strip() for res in re.findall(
            r"<N=[0-3]_execution_result>(.*?)</N=[0-3]_execution_result>",
            inner,
            flags=re.DOTALL
        )]

    def replace_section(txt: str, tag: str) -> str:
        template = "\n".join([
            "<N=0_code><NOTHING></N=0_code>",
            "<N=0_execution_result><NOTHING></N=0_execution_result>",
            "<N=1_code><NOTHING></N=1_code>",
            "<N=1_execution_result><NOTHING></N=1_execution_result>",
        ])
        return re.sub(
            rf"(?s)<{tag}>.*?</{tag}>",
            f"<{tag}>\n{template}\n</{tag}>",
            txt
        )

    # 4 & 5: check primary
    prim_results = extract_results(original, primary_tag)
    prim_errors = [primary_check(r) for r in prim_results]

    if prim_results and all(err is False for err in prim_errors):
        print("4 & 5 <NOTHING> replaced")
        return replace_section(original, primary_tag)

    # 6: check secondary
    sec_results = extract_results(original, secondary_tag)
    sec_errors = [secondary_check(r) for r in sec_results]

    if sec_results and all(err is False for err in sec_errors):
        print("6 <NOTHING> replaced")
        return replace_section(original, secondary_tag)

    # 7: if both have at least one True, replace primary
    if prim_results and sec_results and any(prim_errors) and any(sec_errors):
        print("7 <NOTHING> replaced")
        return replace_section(original, primary_tag)

    # 8: no changes
    return original
