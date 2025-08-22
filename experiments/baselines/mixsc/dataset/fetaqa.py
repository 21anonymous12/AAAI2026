import pandas as pd

from utils.data_loader import download_and_cache_dataset

class FeTaQDataset:
    def __init__(self, cache_dir=None, split='test'):
        self.cache_dir = cache_dir
        self.split = split
        self.dataset = self._load_dataset()

    def _load_dataset(self):
        dataset_url = 'DongfuJiang/FeTaQA'
        local_filename = f'fetaqa_{self.split}.jsonl'
        return download_and_cache_dataset(dataset_url, local_filename, self.cache_dir, self.split)

    def get_item(self, index):
        if index >= len(self.dataset):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.dataset)} entries.")
        
        row = self.dataset.iloc[index]
        table = pd.DataFrame(row.table_array[1:], columns=row.table_array[0])
        question = row['question']
        answer = row['answer']
        question_id = row['feta_id']

        return {
            "table": table,
            "question": question,
            "answer": answer,
            "question_id": question_id
        }

    def __len__(self):
        return len(self.dataset)
    
    
if __name__ == "__main__":
    dataset = FeTaQDataset()
    print(len(dataset))
    print(dataset.get_item(0))