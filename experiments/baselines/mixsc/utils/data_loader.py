import os
import pandas as pd
from datasets import load_dataset

def download_and_cache_dataset(dataset_url, local_filename, cache_dir, split, json_lines=True, data_files=None):
    local_path = os.path.join(cache_dir, local_filename)

    if os.path.exists(local_path):
        return pd.read_json(local_path, lines=json_lines)
    

    dataset = load_dataset(dataset_url, split=split, cache_dir='cache', trust_remote_code=True, data_files=data_files)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    dataset.to_json(local_path)

    return dataset.to_pandas()