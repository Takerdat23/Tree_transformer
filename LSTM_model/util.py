import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        energy = torch.tanh(self.attention(encoder_outputs))  # [batch_size, seq_len, hidden_size]
        attention_weights = F.softmax(self.context_vector(energy), dim=1)  # [batch_size, seq_len, 1]

        context_vector = attention_weights * encoder_outputs  # [batch_size, seq_len, hidden_size]

        return context_vector, attention_weights