# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

## How to Use evaluate.py

```bash
from evaluate import *

pred = 'John , Andy'
target = 'John and Andy'
eval_result = evaluate_all_metrics(pred, target)
print(f"Exact Match : {eval_result['exact_match']}") # A value of 1.0 indicates True, and 0.0 indicates False.
print(f"Fuzzy Matching : {eval_result['Fuzzy_Matching']}")
print(f"F1 score : {eval_result['F1_score']}")
# ---Output---
# Exact Match : 0.0
# Fuzzy Matching : 0.83
# F1 score : 0.66667
```

### If you run the above code, it will print as follows:
```bash
Exact Match : 0.0
Fuzzy Matching : 0.83
F1 score : 0.66667
```


----------------

## Training Datasets for Scheduler and Confidence Checker

You can download these datasets from [this](https://drive.google.com/drive/folders/1kAmC_wJxNI-Be9s9Ug77m4phgLI_ZyOR?usp=sharing) link.

To build this dataset, we leverage three publicly available TableQA datasets: [WikiTQ](https://ppasupat.github.io/WikiTableQuestions/), [TabMWP](https://promptpg.github.io/index.html#home), and [TabFact](https://github.com/wenhuchen/Table-Fact-Checking). Detailed statistics for each source dataset are provided in the table below.
| Dataset  | #Train  | #Val  | #Test  | Main Task              | Table Source                         |
|----------|---------|-------|--------|------------------------|---------------------------------------|
| WikiTQ   | 11,321  | 2,831 | 4,344  | Compositional QA       | Wikipedia (HTML tables)               |
| TabMWP   | 23,059  | 7,686 | 7,686  | Multi-step Math QA     | Curated semi-structured tables        |
| TabFact  | 92,283  | 12,792| 12,779 | Fact Verification      | Wikipedia (infobox-style tables)      |



The table below summarizes the TableQA datasets used for training. **'# table cells'** shows the average number of cells per table in each dataset. **'# CoT'**, **'# PoT'**, and **'# text2SQL'** indicate the number of questions correctly answered by each reasoning method according to the Exact Match metric.
| Dataset | # (T, Q) pairs | # table cells | LLM               | # CoT  | # PoT  | # text2SQL | # Incorrect All |
|---------|----------------|---------------|-------------------|--------|--------|------------|-----------------|
|         |                |               | CodeLLaMA:13B     | 4,832  | 3,813  | 3,284      | 6,378           |
| WikiTQ  |     14,152     |   162.3       | phi4:14B          | 10,034 | 7,877  | 5,117      | 2,578           |
|         |                |               | Qwen2.5-Coder:14B | 8,802  | 8,290  | 5,625      | 2,795           |
|---------|----------------|---------------|-------------------|--------|--------|------------|-----------------|
|         |                |               | CodeLLaMA:13B     | 11,403 | 2,988  | 4,542      | 9,326           |
|  TabMWP |  23,007        | 11.3          | phi4:14B          | 21,592 | 16,806 | 8,464      | 1,004           |
|         |                |               | Qwen2.5-Coder:14B | 21,254 | 17,536 | 9,055      | 1,065           |
|---------|----------------|---------------|-------------------|--------|--------|------------|-----------------|
|         |                |               | CodeLLaMA:13B     | 11,044 | 5,961  | 289        | 7,638           |
| TabFact |  20,729        | 85.6          | phi4:14B          | 17,580 | 12,188 | 10,249     | 1,655           |
|         |                |               | Qwen2.5-Coder:14B | 16,313 | 13,601 | 10,173     | 1,863           |





