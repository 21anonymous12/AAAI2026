import json
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_label(text, tokenizer, model):
    
    
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=2800, return_tensors="pt")

    
    inputs = {key: val.to(device) for key, val in inputs.items()}

    
    with torch.no_grad():
        outputs = model(**inputs)

    
    probabilities = torch.sigmoid(outputs.logits).cpu().numpy().flatten()

    
    return probabilities

