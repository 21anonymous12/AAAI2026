import json
import datasets

class WikiTQDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Wikitablequestions",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(datasets.Value("string")),
                    "table": {
                        "header": datasets.Sequence(datasets.Value("string")),
                        "rows": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                        "name": datasets.Value("string"),
                    },
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        test_data_path = "data/wikitq/wikitq_test.jsonl"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_data_path, "target_split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, target_split):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                data['split'] = target_split
                yield idx, data

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/wikitq.py")
    sample = dataset["test"][0]
    print(sample)