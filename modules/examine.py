from slstm import sLSTMBlock
from transformers import AutoTokenizer

sentences = [
    "đây là một câu",
    "đây là câu thứ hai",
    "câu thứ ba",
    "và câu cuối cùng của đoạn văn"
]
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
input_ids = tokenizer(
    sentences,
    padding="max_length",
    max_length=20,
    return_tensors="pt").input_ids

slstm_block = sLSTMBlock(
    input_size=5000,
    embedding_size=256,
    padding_idx=tokenizer.pad_token_id,
    hidden_size=512
)
output, hidden_size, cell_state, normalizer_state = slstm_block(input_ids)
