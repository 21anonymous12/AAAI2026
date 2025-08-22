import pandas as pd

from utils.data_loader import download_and_cache_dataset

class TabMWPDataset:
    def __init__(self, cache_dir='data', split='test'):
        self.cache_dir = cache_dir
        self.split = split
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        dataset_url = ''
        local_filename = f'tabmwp_{self.split}.json'
        return download_and_cache_dataset(dataset_url, local_filename, self.cache_dir, self.split, json_lines=False).T

    def get_item(self, index):
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.dataset)} entries.")
        
        row = self.dataset.iloc[index]
        table = pd.DataFrame(row.table_for_pd)
        question = row['question'] if not row['choices'] else \
            row['question'] + "\n" + "Here are the choices: " + ", ".join(row['choices'])
        answer = row['answer']

        return {
            "table": table,
            "table_id": index,
            "question": question,
            "answer": answer,
            "question_id": index
        }

    def __len__(self):
        return len(self.dataset)
