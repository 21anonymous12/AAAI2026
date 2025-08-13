from datasets import Dataset, load_dataset
import pandas as pd
import json


class Tablebench_Filtered:
    def __init__(self, dataset, index):
        """데이터와 인덱스를 받아 객체를 초기화합니다."""
        qsubtypes = {'Aggregation',
        'ArithmeticCalculation',
        'Comparison',
        'CorrelationAnalysis',
        'Counting',
        'Domain-Specific',
        'ImpactAnalysis',
        'MatchBased',
        'Multi-hop FactChecking',
        'Multi-hop NumericalReasoing',
        'Ranking',
        'StatisticalAnalysis',
        'Time-basedCalculation',
        'TrendForecasting'}
        # 수정된 부분
        self.dataset = dataset.filter(lambda example: example['qsubtype'] in qsubtypes)
        self.index = index

    def get_table(self):
        data_dict = self.dataset['table'][self.index]
        df = pd.DataFrame(data=data_dict["data"], columns=data_dict["columns"])
        return df

    def get_question(self):
        return self.dataset['question'][self.index]

    def get_answer(self):
        return self.dataset['answer'][self.index]

    def get_qtype(self):
        return self.dataset['qtype'][self.index]

    def get_qsubtype(self):
        return self.dataset['qsubtype'][self.index]

    def display_all(self):
        print("Table DataFrame:")
        df = self.get_table()
        print(df)
        print("\nquestion:", self.get_question())
        print("answer:", self.get_answer())
        print("qtype:", self.get_qtype())
        print("qsubtype:", self.get_qsubtype())
