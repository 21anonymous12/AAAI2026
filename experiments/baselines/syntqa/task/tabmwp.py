import json
import datasets

class TabmwpDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="TabMWP dataset containing table-based math word problems.",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")), 
                    "answer": datasets.Value("string"),
                    "unit": datasets.Value("string"),
                    "table_title": datasets.Value("string"),
                    "table": datasets.Value("string"),
                    "table_for_pd": datasets.Value("string"),
                    "row_num": datasets.Value("int32"),
                    "column_num": datasets.Value("int32"),
                    "solution": datasets.Value("string"),
                    "ques_type": datasets.Value("string"),
                    "ans_type": datasets.Value("string"),
                    "grade": datasets.Value("int32"),
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        test_data_path = "data/tabmwp/tabmwp_test.json"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_data_path, "target_split": "test"},
            )
        ]

    def _generate_examples(self, filepath, target_split):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f) 

        idx = 0
        for ex_id, ex_data in data.items():
            if ex_data.get("split") == target_split:
                yield idx, {
                    "id": ex_id,
                    "question": ex_data.get("question", ""),
                    "choices": ex_data["choices"] if ex_data["choices"] else [],
                    "answer": ex_data.get("answer", ""),
                    "unit": ex_data.get("unit", ""),
                    "table_title": ex_data.get("table_title", ""),
                    "table": ex_data.get("table", ""),
                    "table_for_pd": json.dumps(ex_data.get("table_for_pd", {})),
                    "row_num": ex_data.get("row_num", 0),
                    "column_num": ex_data.get("column_num", 0),
                    "solution": ex_data.get("solution", ""),
                    "ques_type": ex_data.get("ques_type", ""),
                    "ans_type": ex_data.get("ans_type", ""),
                    "grade": ex_data.get("grade", 0),
                    "split": ex_data.get("split", "")
                }
                idx += 1

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/tabmwp.py")
    sample = dataset["train"][0]
    print(sample)