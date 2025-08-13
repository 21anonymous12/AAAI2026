import json, ast
from typing import List, Any

def extract_json(text: str) -> List[Any]:
    """
    임의의 문자열 속에서 {…} / […] 블록을 찾아
    ① json.loads ② ast.literal_eval 순으로 파싱.
    유효하면 Python 객체로 반환.
    """
    objs, stack, start = [], [], None

    for i, ch in enumerate(text):
        if ch in "{[":
            if not stack:         # 새 후보 시작
                start = i
            stack.append(ch)
        elif ch in "}]" and stack:
            opening = stack.pop()
            if (opening, ch) not in {('{', '}'), ('[', ']')}:
                # 괄호 짝이 틀리면 후보 폐기
                stack.clear(); start = None; continue

            if not stack and start is not None:      # 한 덩어리 완성
                candidate = text[start:i+1]
                for loader in (json.loads, ast.literal_eval):
                    try:
                        objs.append(loader(candidate))
                        break                         # 성공했으면 다음 후보로
                    except Exception:
                        continue
                start = None
    return objs


def slice_after_answer(text: str) -> str:
    """
    주어진 문자열에서 'Answer'라는 단어를 찾아
    그 뒤부터 남은 부분을 반환합니다.
    'Answer'가 없으면 원문 그대로 돌려줍니다.
    """
    marker = "Answer"
    idx = text.find(marker)
    if idx == -1:                 # 'Answer'가 없을 때
        return text
    return text[idx + len(marker):].replace("*", "").replace("\n", "").replace(":", "").strip()  # 'Answer' 바로 뒤부터 반환

def filter_dicts_with_keys(items):
    """
    items: mixed 타입이 들어있는 리스트
    반환값: Justification과 Answer 키를 가진 dict들의 리스트
    """
    return [
        d for d in items
        if isinstance(d, dict) and 'Justification' in d and 'Answer' in d
    ]

def extract_answer_from_JA_output(reponce_dataset):
    if {'table', 'question', 'results_all', 'output'}.issubset(reponce_dataset.keys()):
        output_str = reponce_dataset['output']
        try:
            answer = ast.literal_eval(reponce_dataset['output'])
        except:
            try:
                answer = json.loads(output_str)
            except:
                try:
                    answer = filter_dicts_with_keys(extract_json(output_str))[0]
                except:
                    try:
                        answer = slice_after_answer(output_str)
                    except:
                        print("ERROR 1")
                        return output_str
        if type(answer) == str :
            return answer
        elif {'Justification', 'Answer'}.issubset(answer.keys()):
            return answer['Answer']
        else:
            print("ERROR 2")
            return "#LLM FORMAT KEY ERROR!"
