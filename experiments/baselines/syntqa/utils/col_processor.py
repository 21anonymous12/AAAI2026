import re

def col_processor(col_list):
    # 공백, 개행문자, 하이픈 제거
    # col_list = [x.replace("\n", " ").replace("-", "_").strip().replace(" ", "_").replace("___", "_").replace("__", "_").lower() for x in col_list]
    col_list = [x.replace("\n", "_").replace("-", "_")for x in col_list]
    col_list = [re.sub(r"\s+", "_", x)for x in col_list]
    # 특수문자 제거
    col_list = [re.sub(r"[^a-zA-Z0-9_]", "", x) for x in col_list]
    col_list = [x.strip() for x in col_list]
    # 중복 하이픈 제거
    col_list = [re.sub(r"_+", "_", x) for x in col_list]
    col_list = [re.sub(r"_$", "", x) for x in col_list]  # 마지막 _ 제거
    col_list = [re.sub(r"^_", "", x) for x in col_list] 
    col_list = [x.lower() for x in col_list]
    
    # 중복된 컬럼명 처리
    col_list = [f"{col}_{i}" if col_list.count(col) > 1 else col for i, col in enumerate(col_list)]
    
    return col_list

if __name__ == "__main__":
    col_list = ["--asdf_wer  war  \n sddd--"]
    print(col_processor(col_list))
