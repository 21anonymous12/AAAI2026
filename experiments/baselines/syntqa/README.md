# SynTableQA (Custom)

## Installation
```bash
conda create --name syntqa python=3.12
conda activate syntqa
pip install -r requirements.txt
```

## Inference
### TableQA and Text2SQL
```bash
# TableQA
run/{dataset}/{dataset}_tableqa_predict1.sh

# Text2SQL 
run/{dataset}/{dataset}_text_to_sql_predict1.sh
```
If you want to inference with openai model(gpt-4o, etc.), change `python ./run_vllm.py` to `python ./run_openai.py`.

### Selector
```bash
# combine tableqa result and text2sql result
python classifier_custom.py
```
```bash
# run custom selector 
python llm/squall/llm_custom.py
```

### Evaluation
See `notebooks/custom_eval.ipynb`.

## Added Files
```
.
├── run_vllm.py
├── run_openai.py
│ 
├── classifier_custom.py
│
├── task/
│   └── {dataset}.py
├── seq2seq/
│   ├── {dataset}.py
│   └── {dataset}_tableqa.py
├── metric/
│   ├── {dataset}.py
│   └── {dataset}_tableqa.py
├── run/
│   └── {dataset}/
│       ├── {dataset}_tableqa_predict1.sh
│       └── {dataset}_text_to_sql_predict1.sh
└── utils/
    ├── metric.py
    └── processor/
        ├── __init__.py
        ├── table_custom_linearlize.py
        └── table_custom_processor.py

```


# SynTableQA (Original)

Code for 2024 EMNLP Findings

[SynTQA: Synergistic Table-based Question Answering via Mixture of Text-to-SQL and E2E TQA](https://arxiv.org/abs/2409.16682)

Under Construction ...

# Experiments

`run.py` is the main code to finetune and inference pretrained T5 and Omnitab models. `classifier.py` train the random forest classier to select more correct answer from predictions (i.e., csv files in `predict` folder) from end-to-end table QA model and Text-to-SQL model. A `data` folder should be prepared in the root dir which clones from `Squall` and `WikiSQL` projects. 

# Contact

For any issues or questions, kindly email us at: Siyue Zhang (siyue001@e.ntu.edu.sg).

# Citation

```
@inproceedings{zhang-etal-2024-syntqa,
    title = "{S}yn{TQA}: Synergistic Table-based Question Answering via Mixture of Text-to-{SQL} and {E}2{E} {TQA}",
    author = "Zhang, Siyue  and
      Luu, Anh Tuan  and
      Zhao, Chen",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.131",
    pages = "2352--2364",
    abstract = "Text-to-SQL parsing and end-to-end question answering (E2E TQA) are two main approaches for Table-based Question Answering task. Despite success on multiple benchmarks, they have yet to be compared and their synergy remains unexplored. In this paper, we identify different strengths and weaknesses through evaluating state-of-the-art models on benchmark datasets: Text-to-SQL demonstrates superiority in handling questions involving arithmetic operations and long tables; E2E TQA excels in addressing ambiguous questions, non-standard table schema, and complex table contents. To combine both strengths, we propose a Synergistic Table-based Question Answering approach that integrate different models via answer selection, which is agnostic to any model types. Further experiments validate that ensembling models by either feature-based or LLM-based answer selector significantly improves the performance over individual models.",
}
```
