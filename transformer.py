import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig

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
        self.vocab = vocab

        self.embedding = nn.Embedding(vocab.size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.encoder = BertEncoder(config)

        self.fc = nn.Linear(d_model, vocab.total_tags)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids: torch.Tensor):
        attention_mask = torch.not_equal(input_ids, self.vocab.padding_idx).long()
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        attention_mask = (1.0 - attention_mask) * -10000
        logits = self.embedding(input_ids)
        logits = self.layer_norm(logits)
        logits = self.encoder(logits, attention_mask)[0]
        logits = self.fc(logits)

        return logits
