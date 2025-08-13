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
