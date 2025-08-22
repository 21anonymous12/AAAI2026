import json
import datasets

class TableBenchDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="Tablebench",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "qtype": datasets.Value("string"),
                    "qsubtype": datasets.Value("string"),
                    "table": {
                        "columns": datasets.Sequence(datasets.Value("string")),
                        "data": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                    },
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                    "chart_type": datasets.Value("string"),
                    "split": datasets.Value("string"),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        test_data_path = "data/tablebench/tablebench_train.jsonl"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": test_data_path, "target_split": "test"},
            )
        ]

    def _generate_examples(self, filepath, target_split):
        qsubtypes = { 
            'Aggregation',
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
            'TrendForecasting'
        }
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                if data['qsubtype'] not in qsubtypes:
                    continue
                data['split'] = target_split
                yield idx, data

if __name__ == '__main__':
    from datasets import load_dataset
    dataset = load_dataset("./task/tablebench.py")
    sample = dataset["test"][0]
    print(sample)