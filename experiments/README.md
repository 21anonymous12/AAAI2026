# MATA: Multi-Agent Framework for Reliable and Flexible Table Question Answering

## How to Use evaluate.py

```bash
from evaluate import *

pred = 'John , Andy'
target = 'John and Andy'
eval_result = evaluate_all_metrics(pred, target)
print(f"Exact Match : {eval_result['exact_match']}")
print(f"Fuzzy Matching : {eval_result['Fuzzy_Matching']}")
print(f"F1 score : {eval_result['F1_score']}")
# ---Output---
# Exact Match : False
# Fuzzy Maching : 0.83
# F1 score : 0.67
```

Running this script will print 
'''bash
Exact Match : False
Fuzzy Maching : 0.83
F1 score : 0.67
'''
