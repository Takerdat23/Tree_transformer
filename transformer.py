import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertConfig

from data_utils.vocab import Vocab

class Transformer(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 num_layers: int, 
                 d_ff: int, 
                 vocab: Vocab):
        super().__init__()

        config = BertConfig(
            vocab_size=vocab.size,
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            pad_token_id=vocab.padding_idx,
            intermediate_size=d_ff
        )

        self.embedding = nn.Embedding(vocab.size, d_model)
        self.self_attention = BertSelfAttention(config)

        self.fc = nn.Linear(d_model, vocab.total_tags)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids: torch.Tensor):
        logits = self.embedding(input_ids)
        logits = self.self_attention(logits)[0]
        logits = self.fc(logits)
        logits = self.softmax(logits)
        outputs = logits.argmax(dim=-1)

        return outputs, logits
