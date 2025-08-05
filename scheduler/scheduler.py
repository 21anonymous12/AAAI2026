
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, MobileBertModel
from typing import List, Dict


class MobileBertWithFeatures(nn.Module):
    def __init__(self, num_extra: int, dropout: float = 0.1):
        super().__init__()
        self.bert = MobileBertModel.from_pretrained("google/mobilebert-uncased")
        hidden = self.bert.config.hidden_size          # 512
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + num_extra, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)                         
        )

    def forward(self, input_ids, attention_mask, extra_feats):
        cls = self.bert(input_ids, attention_mask).last_hidden_state[:, 0]  # [B, 512]
        x = torch.cat([cls, extra_feats], dim=1)                            # [B, 512+E]
        logits = self.head(x)                                               # [B, 2]
        return logits              