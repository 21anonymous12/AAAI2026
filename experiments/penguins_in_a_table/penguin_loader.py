import pandas as pd
import json
from io import StringIO


class Penguins_in_a_table:
    def __init__(self, dataset, index):
        """데이터와 인덱스를 받아 객체를 초기화합니다."""
        self.dataset = dataset
        self.index = index

    # Getter 메서드들
    def get_table(self):
        """테이블(DataFrame) 반환"""
        df = pd.read_csv(StringIO(self.dataset[f'{self.index}']['Table']))
        return df

    def get_question(self):
        """주어진 문장(statement) 반환"""
        return self.dataset[f'{self.index}']['Question']

    def get_option(self):
        """주어진 문장(statement) 반환"""
        return self.dataset[f'{self.index}']['Options']

    def get_target(self):
        """정답 (answer) 반환"""
        return self.dataset[f'{self.index}']['target']

    def get_answer_extract(self):
        """정답 (answer) 반환"""
        return self.dataset[f'{self.index}']['answer_extract']

    def display_all(self):
        """모든 데이터를 출력"""
        print("Table DataFrame:")
        df = self.get_table()
        print(df)
        print("\Question:", self.dataset[f'{self.index}']['Question'])
        print("\nOption:", self.dataset[f'{self.index}']['Options'])
        print("\ntarget:", self.dataset[f'{self.index}']['target'])
        print("\nanswer_extract:", self.dataset[f'{self.index}']['answer_extract'])


#메서드 사용 예시
#fetaqa = FeTaQA(df, index)
#table = fetaqa.get_table_array()
#query = fetaqa.get_question()
#answers = fetaqa.get_answer()