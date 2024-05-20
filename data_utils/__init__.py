import torch

def pad_tensor(tensors: list[torch.Tensor], pad_value: int = 0):
    max_len = max([tensor.shape[-1] for tensor in tensors])
    padded_tensors = []
    for tensor in tensors:
        if tensor.dim() == 2:
            pad_tensor = torch.zeros((1, max_len - tensor.shape[-1], )).fill_(pad_value).long()
        else:
            pad_tensor = torch.zeros((max_len - tensor.shape[-1], )).fill_(pad_value).long()
        padded_tensor = torch.cat([tensor, pad_tensor], dim=-1)
        padded_tensors.append(padded_tensor)
    padded_tensors = torch.cat(padded_tensors)

    return padded_tensors

def collate_fn(items: list) -> torch.Tensor:
    batch_input_ids = []
    batch_tags = []
    for item in items:
        batch_input_ids.append(item["input_ids"].unsqueeze(0))
        batch_tags.append(item["tags"])

    batch_input_ids = pad_tensor(batch_input_ids)
    batch_tags = pad_tensor(batch_tags)

    return batch_input_ids, batch_tags