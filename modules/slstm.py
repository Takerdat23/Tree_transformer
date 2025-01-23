import torch
from torch import nn
from torch.nn import functional as F

class DiagLinear(nn.Module):
    def __init__(self, input_features: int, output_features: int, bias: bool = True):
        assert input_features == output_features, f"input_features and output_features must be the same, get input_feautures {input_features} and output_features {output_features}"
        self._weights = torch.zeros((input_features, ) if bias else (input_features+1, ))
        self._bias = torch.zeros((output_features, )) if bias else None
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self._weights)

    def forward(self, features: torch.Tensor):
        """
            features: (*, dim)
        """
        weights = torch.diag_embed(self._weights)
        output = F.linear(features, weights, self._bias)

        return output
    
class PFLinear(nn.Module):
    def __init__(self, input_features: int, PF: float, bias: bool = True):
        output_features = int(PF*input_features)

        self._weights = torch.zeros((input_features, output_features))
        self._bias = torch.zeros((output_features, )) if bias else None

    def init_weights(self):
        nn.init.xavier_uniform_(self._weights)

    def forward(self, features: torch.Tensor):
        return F.linear(features, self._weights, self._bias)

class sLSTMCell(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        
        self.forget_input = DiagLinear(hidden_size, hidden_size, bias=False)
        self.forget_hidden = DiagLinear(hidden_size, hidden_size, bias=True)
        
        self.input_input = DiagLinear(hidden_size, hidden_size, bias=False)
        self.input_hidden = DiagLinear(hidden_size, hidden_size, bias=True)

        self.cell_input = DiagLinear(hidden_size, hidden_size, bias=False)
        self.cell_hidden = DiagLinear(hidden_size, hidden_size, bias=True)

        self.output_input = DiagLinear(hidden_size, hidden_size, bias=False)
        self.output_hidden = DiagLinear(hidden_size, hidden_size, bias=True)

    def forward(self, 
                inputs: torch.Tensor, 
                hidden_state: torch.Tensor, 
                cell_state: torch.Tensor,
                normalizer_state: torch.Tensor):
        """
            inputs: (batch size, hidden size)
            hidden_state: (batch size, hidden size)
            cell_state: (batch size, hidden size)
            normalizer_state: (batch size, hidden size)
        """

        candidate_input = self.input_input(inputs) + self.input_hidden(hidden_state)
        candidate_forget = self.forget_input(inputs) + self.forget_hidden(hidden_state)
        candidate_cell = self.cell_hidden(inputs) + self.cell_hidden(hidden_state)
        candidate_output = self.output_input(inputs) + self.output_hidden(hidden_state)

        output = F.softmax(candidate_output, dim=-1)
        input = torch.exp(candidate_input)
        forget = torch.logical_or(torch.exp(candidate_forget), F.sigmoid(candidate_forget))
        cell = F.tanh(candidate_cell)

        cell_state = forget*cell_state + input*cell
        normalizer_state = forget*normalizer_state + input
        candidate_hidden = cell_state*normalizer_state.inverse()
        hidden_state = output*candidate_hidden

        return output, hidden_state, cell_state, normalizer_state

class MLP(nn.Module):
    def __init__(self, hidden_size, PF: float):
        super().__init__()

        self.left_up_proj = PFLinear(hidden_size, PF)
        self.right_up_proj = PFLinear(hidden_size, PF)
        self.down_proj = PFLinear(hidden_size, PF**-1)

    def forward(self, features: torch.Tensor):
        features = self.left_up_proj(features) * F.gelu(self.right_up_proj(features))
        features = self.down_proj(features)

        return features
    
class sLSTMBlock(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, padding_idx: int, hidden_size: int):
        super().__init__()

        self.embedding = nn.Embedding(input_size, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.proj = nn.Linear(embedding_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.conv4 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=4)

        self.slstm = sLSTMCell(hidden_size)
        self.group_norm = nn.GroupNorm(num_groups=4, num_channels=4*hidden_size)
        self.mlp = MLP(hidden_size, PF=4/3)

    def forward(self, input_ids):
        input_embs = self.embedding(input_ids)
        input_embs = self.dropout(self.proj(input_embs))
        input_embs = self.dropout(input_embs)

        input_embs = input_embs.permute((0, -1, 1))
        input_embs = self.conv4(input_embs)
        input_embs = input_embs.permute((0, -1, 1))
        bs, seq_len, hidden_size = input_embs.shape

        # cell state and hidden state initialization
        cell_state = torch.zeros((bs, seq_len, hidden_size))
        hidden_state = torch.zeros((bs, seq_len, hidden_size))
        # normalizer state initialization
        normalizer_state = torch.zeros((bs, seq_len, hidden_size))
        normalizer_state = nn.init.xavier_uniform_(normalizer_state)
        for ith in range(seq_len):
            input_emb = input_embs[ith]

            input_emb = self.layer_norm(input_emb)
            input_emb = F.silu(self.conv4(input_emb))
            output, hidden_state, cell_state, normalizer_state = self.slstm(
                input_emb,
                hidden_state,
                cell_state,
                normalizer_state
            )
            group_inputs = torch.cat([
                output, hidden_state, cell_state, normalizer_state
            ], dim=-1)
            group_embs = self.group_norm(group_inputs)
            group_embs = self.mlp(group_embs)
            # split the output
            output = group_embs[:, :hidden_size] # we will not use this for encoder LSTM method
            hidden_state = group_embs[:, hidden_size:2*hidden_size]
            cell_state = group_embs[:, 2*hidden_size:3*hidden_size]
            normalizer_state = group_embs[:, 3*hidden_size:4*hidden_size]

        return output, hidden_size, cell_state, normalizer_state
