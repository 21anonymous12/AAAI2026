error_text2sql_pattern = ["<NOTHING>",'Error:', '[nan]']



error_text2sql_match_pattern = ["<NOTHING>", "code", "'code'", '[]']


def check_text2sql_error(text):
    if len(text) > 3000:
        return False
    for pattern in error_text2sql_pattern:
        if pattern in text:
            return False
    for pattern in error_text2sql_match_pattern:
        if text == pattern:
            return False
    return text