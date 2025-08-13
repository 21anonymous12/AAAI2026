# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

## How to Use evaluate.py

```bash
from evaluate import *

pred = 'John , Andy'
target = 'John and Andy'
eval_result = evaluate_all_metrics(pred, target)
print(f"Exact Match : {eval_result['exact_match']}") # The value 1.0 means True, and 0.0 means False.
print(f"Fuzzy Matching : {eval_result['Fuzzy_Matching']}")
print(f"F1 score : {eval_result['F1_score']}")
# ---Output---
# Exact Match : 0.0
# Fuzzy Maching : 0.83
# F1 score : 0.66667
```

### Running this script will print as follows:
```bash
Exact Match : 0.0
Fuzzy Maching : 0.83
F1 score : 0.66667
```


----------------

## Training Datasets for Scheduler And Confidence Checker

You can download these from [this](https://drive.google.com/drive/folders/1kAmC_wJxNI-Be9s9Ug77m4phgLI_ZyOR?usp=sharing) link.
