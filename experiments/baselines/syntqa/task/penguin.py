import json
import datasets

class PenguinDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Penguin",
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "options": datasets.Value("string"),
                    "table": datasets.Value("string"),
                    "target": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        test_data_path = "data/penguin/penguin_test.json"
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
            yield idx, {
                "id": idx,
                "question": ex_data["Question"],
                "options": ex_data["Options"],
                "table": ex_data["Table"],
                "target": ex_data["target"],
                "answer": ex_data["answer_extract"],
                "split": target_split,
            }
            idx += 1


if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/penguin.py")
    sample = dataset["test"][0]
    print(sample)