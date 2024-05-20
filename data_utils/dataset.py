from torch.utils.data import Dataset

from data_utils.vocab import Vocab, preprocess_sentence
import json

class PhoNERT_Dataset(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        self.vocab = vocab
        self.annotations = self.load_annotations(path)

    def load_annotations(self, path: str):
        annotations = []
        lines = open(path).readlines()
        for line in lines:
            annotation = json.loads(line.strip())
            words = annotation["words"]
            annotations.append({
                "sentence": " ".join(words),
                "tags": annotation["tags"]
            })

        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        annotation = self.annotations[idx]
        inputs_ids = self.vocab.encode_sentence(
            preprocess_sentence(annotation["sentence"])
        )
        tags = self.vocab.encode_tag(annotation["tags"])

        return {
            "input_ids": inputs_ids,
            "tags": tags
        }
