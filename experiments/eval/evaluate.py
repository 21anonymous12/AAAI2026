# fuzzywuzzy
from fuzzywuzzy import fuzz

# BERTScore
from bert_score import score

import math
import re
import numpy as np
from tqdm import tqdm
import pandas as pd
import ast

def text_to_list(text): # for wikiTQ
    """
    텍스트 형태의 리스트를 실제 리스트로 변환.

    :param text: 문자열 (예: "['Siim Ennemuist', 'Andri Aganits']")
    :return: 리스트 (예: ['Siim Ennemuist', 'Andri Aganits'])
    """
    return ast.literal_eval(text)


def normalize_fraction(text):
    """
    텍스트를 분수 형태로 표준화 (텍스트 기반 처리).
    :param text: str, 입력 텍스트
    :return: str, 표준화된 분수 텍스트
    """
    # LaTeX 분수 표현을 정규식으로 변환
    latex_fraction_pattern = r"\\frac{(\d+)}{(\d+)}"
    match = re.search(latex_fraction_pattern, text)
    if match:
        numerator, denominator = match.groups()
        return f"{numerator}/{denominator}"
    return text  # LaTeX 형식이 아니면 원본 반환


def normalize_time_format(text):
    """
    시간 표현을 정규화하여 포함 비교 가능하게 처리.
    P.M. 및 A.M.과 같은 특수한 시간 표현을 처리.
    """
    return text.replace("P.M.", "P.M").replace("A.M.", "A.M").strip()


def parse_numeric_value(text):
    """
    텍스트를 숫자 값으로 파싱하려고 시도합니다.
    분수, 소수, 퍼센트, LaTeX 분수 등을 처리합니다.
    :param text: str, 입력 텍스트
    :return: float 또는 None, 파싱된 숫자 값 또는 실패 시 None
    """
    text = text.strip()
    text = text.replace(",", "")  # 쉼표 제거

    # 퍼센트 처리
    if text.endswith('%'):
        try:
            value = float(text[:-1]) / 100.0
            return value
        except ValueError:
            pass

    # LaTeX 분수 처리
    match = re.match(r"\\frac{(-?\d+)}{(-?\d+)}", text)
    if match:
        numerator, denominator = match.groups()
        try:
            value = float(numerator) / float(denominator)
            return value
        except (ValueError, ZeroDivisionError):
            pass

    # 일반 분수 처리
    if '/' in text:
        try:
            numerator, denominator = text.split('/')
            value = float(numerator) / float(denominator)
            return value
        except (ValueError, ZeroDivisionError):
            pass

    # 소수 또는 정수 처리
    try:
        value = float(text)
        return value
    except ValueError:
        pass

    # 파싱 실패 시 None 반환
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

    # 퍼센트 비교
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

    # $, %, , 제거 후 비교
    ans_dollar = str(ans).replace("$", "").replace(",", "").replace("%", "").strip()
    pred_dollar = str(pred).replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        if dynamic_isclose(pred_dollar, ans_dollar):
            return True
    except:
        pass

    # parse_numeric_value 후 비교
    try:
        pred_val = parse_numeric_value(pred)
        ans_val = parse_numeric_value(ans)
        if isinstance(pred_val, float) and isinstance(ans_val, float):
            if dynamic_isclose(pred_val, ans_val):
                return True
    except:
        pass

    # 문자열 포함 비교는 진짜 텍스트일 때만
    if isinstance(pred, str) and isinstance(ans, str):
        try:
            float(pred)
            float(ans)
            # 둘 다 숫자처럼 보이면 포함 비교는 생략
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



def compute_fuzzy_matching(pred, ans):
    # 2) Fuzzy Matching
    if check_exact_match(pred, ans) == True:
        return 1.0
    else:
        pred_str = str(pred).lower().strip()
        ans_str = str(ans).lower().strip()
        fuzzy_val = fuzz.ratio(pred_str, ans_str) / 100.0 
        return fuzzy_val


def compute_f1(prediction, ground_truth):
    """
    SQuAD-style token-level F1 score
    """
    if check_exact_match(prediction, ground_truth) == True:
        return 1.0
    else:
        pred_tokens = str(prediction).lower().split()
        gt_tokens = str(ground_truth).lower().split()
        common = set(pred_tokens) & set(gt_tokens)
        num_same = len(common)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1


def evaluate_all_metrics(pred, ans):
    # 1) Exact match
    exact = 1.0 if check_exact_match(pred, ans) else 0.0

    # 2) Fuzzy Matching
    
    fuzzy_val = compute_fuzzy_matching(pred, ans)


    # 3) Token-level F1

    f1_score_val = compute_f1(pred, ans)


    return {
        'exact_match': exact,
        'Fuzzy_Matching': round(fuzzy_val, 5),
        'F1_score': round(f1_score_val, 5),
    }
