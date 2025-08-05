error_PoT_pattern = ["Can only use .str accessor with string values!", ".str accessor", "unsupported operand", "Reindexing only valid", "Series is ambiguous", "arg must be", "string values!", "Non-numeric values", "Non-numeric", "<NOTHING>","nameerror","typeerror","<unknown>","keyerror","syntaxerror","indexerror", "valueerror", "dtype='int64'", "no overlapping index names", "cannot reindex", "duplicate labels", "bytes-like object", "is not defined", "argument must be a string or a real number", "'NAType'", "first argument must be an iterable of pandas objects", "you passed an object of type", "Error: ", "cannot be concatenated", "cannot join with no overlapping index names","must be of the same length","does not match format","scalar, tuple, or list","NoneType","must be a string","None of [Index(", "invalid literal for ", "out of bounds for", "has no attribute", "invalid syntax", "(<string>, line", "dtype: object", "not supported between instances", "list index out of range",  "is out-of-bounds", "unhashable type:", "None of [Index", "invalid index", "index out of range", "not convert string to float", "not found in axis", "The truth value of a Series is ambiguous.", "Length mismatch: Expected ", "get argmax of an empty sequence", "Unable to parse string", 'arg is an empty sequence', 'No unique match found', '(expected ', 'not enough values to unpack', 'expected hh:mm:ss format before', ' object is not iterable', "doesn't match format", "dtype: int64", "can only concatenate str", "unsupported operand type(s) for", "pattern contains no capture groups", "nothing to repeat at position", "takes no keyword arguments", "Something has gone wrong, please report a bug at", "No valid entry dates", "attempt to get argmin", "No specific type mentioned", "column not found in data", "arg must be a list, tuple, 1-d array, or Series", "did not contain a loop with signature matching types", "<class ", "cannot convert float", "nomination found for", "No valid attendance data available", "Length of values", "not match length of index", "Invalid comparison between", "Data not available in the provided DataFrame", "unexpected keyword argument", "not supported for the", "No specific issue range found for", "does not support strftime", "No valid", "unconverted data remains when parsing with format", "could not be broadcast", "operands could not", "object do not match", "Unalignable boolean Series", "Boolean array expected", "Series.replace cannot use dict-like to_replace and non-None value", "Expected value of kwarg", "No matching driver combination found", "unit must not be specified", "if the input contains a str", "integer or boolean arrays are valid indices"]



error_PoT_match_pattern = ["<NOTHING>","code", "'code'", "'index'", "'ans'", "nan", "None", "[nan]", "ans", "$nan", "[]"]


def check_PoT_error(text):
    if len(text) > 3000:
        return False
    for pattern in error_PoT_pattern:
        if pattern in text:
            return False
    for pattern in error_PoT_match_pattern:
        if text == pattern:
            return False
    return text