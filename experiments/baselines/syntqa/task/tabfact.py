import json
import datasets

class TabfactDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Tabfact",
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "table": {
                        "id": datasets.Value("string"),
                        "header": datasets.Sequence(datasets.Value("string")),
                        "rows": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                        "caption": datasets.Value("string"),
                    },
                    "statement": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        test_data_path = "data/tabfact/tabfact_test.jsonl"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_data_path, "target_split": "test"},
            )
        ]

    def _generate_examples(self, filepath, target_split):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                data['split'] = target_split
                yield idx, data

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/tabfact.py")
    sample = dataset["test"][0]
    print(sample)