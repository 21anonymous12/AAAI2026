import pandas as pd
import json

from utils.data_loader import download_and_cache_dataset

class TableBenchDataset:
    def __init__(self, cache_dir='data', split='test'):
        self.cache_dir = cache_dir
        self.split = split
        self.dataset = self._load_dataset()
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
        self.dataset_filtered = self.dataset[self.dataset['qsubtype'].isin(qsubtypes)]

    def _load_dataset(self):
        dataset_url = 'Multilingual-Multimodal-NLP/TableBench'
        local_filename = f'tablebench_{self.split}.jsonl'
        return download_and_cache_dataset(dataset_url, local_filename, self.cache_dir, self.split, data_files="TableBench.jsonl")

    def get_item(self, index):
        if index >= len(self.dataset_filtered):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.dataset_filtered)} entries.")
        
        row = self.dataset_filtered.iloc[index]
        table_dict = row['table']
        table = pd.DataFrame(data=table_dict['data'], columns=table_dict['columns'])
        question = row['question']
        answer = row['answer']

        return {
            "table": table,
            "table_id": row['id'],
            "question": question,
            "answer": answer,
            "question_id": index
        }

    def __len__(self):
        return len(self.dataset_filtered)
