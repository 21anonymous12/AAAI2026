
import math
import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import ast

def text_to_list(text): 
    """
    Convert a text-formatted list into an actual list.

    :param text: A string (e.g., "['Siim Ennemuist', 'Andri Aganits']")
    :return: A list (e.g., ['Siim Ennemuist', 'Andri Aganits'])
    """
    return ast.literal_eval(text)


def normalize_fraction(text):
    """
    Standardize the text into fractional format (text-based processing).
    :param text: str, input text  
    :return: str, standardized fractional text  
    """
    
    latex_fraction_pattern = r"\\frac{(\d+)}{(\d+)}"
    match = re.search(latex_fraction_pattern, text)
    if match:
        numerator, denominator = match.groups()
        return f"{numerator}/{denominator}"
    return text  


def normalize_time_format(text):
    """
    Normalize time expressions to enable inclusive comparison.
    Handle special time formats such as "P.M." and "A.M.".
    """
    return text.replace("P.M.", "P.M").replace("A.M.", "A.M").strip()


def parse_numeric_value(text):
    """
    Attempt to parse the text into a numeric value.
    Handles fractions, decimals, percentages, LaTeX-style fractions, and more.
    :param text: str, input text  
    :return: float or None, the parsed numeric value or None if parsing fails  
    """
    text = text.strip()
    text = text.replace(",", "") 

    # Handle percentage values.
    if text.endswith('%'):
        try:
            value = float(text[:-1]) / 100.0
            return value
        except ValueError:
            pass

    # Handle LaTeX-style fractions.
    match = re.match(r"\\frac{(-?\d+)}{(-?\d+)}", text)
    if match:
        numerator, denominator = match.groups()
        try:
            value = float(numerator) / float(denominator)
            return value
        except (ValueError, ZeroDivisionError):
            pass

    # Handle regular fractions.
    if '/' in text:
        try:
            numerator, denominator = text.split('/')
            value = float(numerator) / float(denominator)
            return value
        except (ValueError, ZeroDivisionError):
            pass

    # decimals or integers.
    try:
        value = float(text)
        return value
    except ValueError:
        pass

    # Return None if parsing fails.
    return text

def dynamic_isclose(a, b):
    try:
        a = float(a)
        b = float(b)
        if max(abs(a), abs(b)) < 500:
            a_val = round(float(a), 3)
            b_val = round(float(b), 3)
            return math.isclose(a_val, b_val, rel_tol=1e-3, abs_tol=1e-3)
        else:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    except:
        return False


def check_exact_match(pred, ans):
    if (isinstance(pred, str) and pred.strip() == "") or (isinstance(ans, str) and ans.strip() == ""):
        return False
    if isinstance(pred, list) and len(pred) == 0:
        return False
    if isinstance(ans, list) and len(ans) == 0:
        return False

    try:
        if dynamic_isclose(pred, ans):
            return True
    except:
        pass


    if isinstance(pred, str) and "%" in pred:
        try:
            if dynamic_isclose(float(pred.replace("%", "")) / 100.0, ans):
                return True
        except:
            pass
        try:
            if dynamic_isclose(float(pred.replace("%", "")), ans):
                return True
        except:
            pass

    if isinstance(ans, str) and "%" in ans:
        try:
            if dynamic_isclose(float(ans.replace("%", "")) / 100.0, pred):
                return True
        except:
            pass
        try:
            if dynamic_isclose(float(ans.replace("%", "")), pred):
                return True
        except:
            pass


    ans_dollar = str(ans).replace("$", "").replace(",", "").replace("%", "").strip()
    pred_dollar = str(pred).replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        if dynamic_isclose(pred_dollar, ans_dollar):
            return True
    except:
        pass


    try:
        pred_val = parse_numeric_value(pred)
        ans_val = parse_numeric_value(ans)
        if isinstance(pred_val, float) and isinstance(ans_val, float):
            if dynamic_isclose(pred_val, ans_val):
                return True
    except:
        pass


    if isinstance(pred, str) and isinstance(ans, str):
        try:
            float(pred)
            float(ans)

        except:
            pred_lower = pred.lower().strip()
            ans_lower = ans.lower().strip()
            if pred_lower == ans_lower:
                return True

            pred_clean = pred_lower.replace("'", "").replace("the", "").strip()
            ans_clean = ans_lower.replace("'", "").replace("the", "").strip()
            if pred_clean == ans_clean:
                return True

            pred_fraction = normalize_fraction(pred_lower)
            ans_fraction = normalize_fraction(ans_lower)
            if pred_fraction == ans_fraction:
                return True

            pred_time = normalize_time_format(pred_lower)
            ans_time = normalize_time_format(ans_lower)
            if pred_time == ans_time:
                return True

            pred_bar = pred_lower.replace("-", "–").strip()
            ans_bar = ans_lower.replace("-", "–").strip()
            if pred_bar == ans_bar:
                return True

            pred_double = pred_lower.replace('"', "").replace(".", "").strip()
            ans_double = ans_lower.replace('"', "").replace(".", "").strip()
            if pred_double == ans_double:
                return True

            pred_the = pred_double.replace("the", "").strip()
            ans_the = ans_double.replace("the", "").strip()
            if pred_the == ans_the:
                return True

    return False
