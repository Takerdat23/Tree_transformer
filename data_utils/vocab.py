import torch

from collections import Counter
import json
from typing import List, Union
import re

def preprocess_sentence(sentence: Union[str, List[str]]) -> List[str]:
    if isinstance(sentence, list):
        sentence = " ".join(sentence)
    
    sentence = sentence.lower()    
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)

    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()
    
    return tokens

class Vocab(object):
    """
        Defines a vocabulary object that will be used to numericalize a field.
    """
    def __init__(self, paths: list[str]):

        self.padding_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        freqs, tags, self.max_sentence_length = self.make_vocab(paths)
        self.i2tags = {ith: tag for ith, tag in enumerate(list(tags), 1)}
        self.i2tags[0] = self.padding_token
        self.tags2i = {tag: ith for ith, tag in enumerate(list(tags), 1)}
        counter = freqs.copy()

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        itos = specials
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < 5:
                break
            itos.append(word)

        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}

        self.specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

    def make_vocab(self, paths):
        max_sentence_length = 0
        freqs = Counter()
        tags = set()
        for path in paths:
            lines = open(path).readlines()
            for line in lines:
                annotation = json.loads(line)
                words = preprocess_sentence(annotation["words"])
                if len(words) > max_sentence_length:
                    max_sentence_length = len(words)
                tags.update(annotation["tags"])
                freqs.update(words)

        return freqs, tags, max_sentence_length

    def encode_sentence(self, sentence: List[str]) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        vec = torch.ones(len(sentence) + 2)
        for i, token in enumerate([self.bos_token] + sentence + [self.eos_token]):
            vec[i] = self.stoi[token] if token in self.stoi else self.unk_idx
        return vec.long()

    def encode_tag(self, tags: list[str]) -> torch.Tensor:
        """ Turn tags into a vector of indices and a question length """
        vec = torch.zeros((len(tags), )).long()
        for ith in range(len(tags)):
            tag = tags[ith]
            vec[ith] = self.tags2i[tag]

        return vec.long()

    def decode_sentence(self, sentence_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in sentence_vecs:
            sentence = " ".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                sentences.append(sentence)
            else:
                sentences.append(sentence.strip().split())

        return sentences

    def decode_tag(self, tag_vecs: torch.Tensor, mask: torch.Tensor) -> List[str]:
        '''
            tag_vecs: (bs, max_length)
        '''
        batch_tags = []
        for tag_vec in tag_vecs:
            tags = tag_vec.tolist()
            tags = [self.i2tags[tag] for tag in tags]
            batch_tags.append(tags)

        return batch_tags

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False

        return True

    def __len__(self):
        return len(self.itos)
    
    @property
    def total_tags(self) -> int:
        return len(self.tags2i) + 1
    
    @property
    def size(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos.values()) if sort else v.itos.values()
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
