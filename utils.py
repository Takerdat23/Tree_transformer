from __future__ import print_function
import numpy as np
import random
import json
import os
import re
import sys
import torch
from tqdm import tqdm
import operator
import torch.autograd as autograd
from nltk.corpus import stopwords
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd 
import time
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import json 
from underthesea import word_tokenize


def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data


def write_json(filename,data):
    with open(filename, 'w', encoding='utf8') as fp:
        json.dump(data, fp)


def make_save_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

#for aspect-based tasks




def prepare_data(file_path):
    df = pd.read_csv(file_path,  encoding = 'utf8')

    # remove nan
    df = df.dropna()
    df = df.reset_index(drop=True)

    dataset = []

    texts = df['content'].tolist()
    spans = df['index_spans'].tolist()

    # convert spans to binary representation

    for text , span in zip(texts,  spans):
        data_dict = {}
        binary_span = []
        span = span.split(' ')
        for s in span:
            if s == 'O':
                binary_span.append(0)
            else:
                binary_span.append(1)
        data_dict["text"] = text
        data_dict["span"] = binary_span
        dataset.append(data_dict)
        
    return dataset

# Dataloader function




def cc(arr, no_cuda=False):
    if no_cuda:
        return torch.from_numpy(np.array(arr))
    else:
        return torch.from_numpy(np.array(arr)).cuda()






class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        inputs = [example["text"] for example in batch]
        spans = [example["span"] for example in batch]


        encoded_batch = [self.tokenizer.encode(sentence) for sentence in inputs]

        # Get the maximum sequence length
        max_length = 128

        # Pad and truncate the sequences
        for encoded in encoded_batch:
            encoded.pad(max_length)
            encoded.truncate(max_length)

        # Convert the sequences to numpy arrays
        input_ids = torch.tensor([encoded.ids for encoded in encoded_batch])
        attention_mask = torch.tensor([encoded.attention_mask for encoded in encoded_batch])

 
     
        for span in spans:
            if len(span) < max_length:
                spans.append(span + [0] * (max_length - len(span)))
            else:
                spans.append(span[:max_length])

        spans = torch.tensor(spans)
       

      
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "spans": spans}


class data_utils():
    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.no_cuda = args.no_cuda
        self.train_path = args.train_path

        df_train = pd.read_csv(args.train_path,  encoding = 'utf8') 
        df_train = df_train.dropna()
        df_train = df_train.reset_index(drop=True)
  
     
        
        if os.path.exists(os.path.join(args.model_dir,"vocab.json" )) and os.path.exists(os.path.join(args.model_dir,"merges.txt" )): 
            # self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.tokenizer = ByteLevelBPETokenizer.from_file( os.path.join(args.model_dir,"vocab.json" ), os.path.join(args.model_dir,"merges.txt" ))
        else: 
            print("No Tokenizer found")
            # self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            
            tokenizer = ByteLevelBPETokenizer()

            tokenizer.train_from_iterator(df_train["content"], vocab_size=30000, min_frequency=2,
                              special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.save_model(args.model_dir)
            self.tokenizer = ByteLevelBPETokenizer.from_file( os.path.join(args.model_dir,"vocab.json" ), \
                                                             os.path.join(args.model_dir,"merges.txt" ))

          
        data_collator = DataCollator(self.tokenizer)
       
        dataset = prepare_data(args.train_path)
        val_dataset =  prepare_data(args.valid_path)
        
        self.train_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
        self.val_loader =DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)


        if args.test : 
        
            test_dataset = prepare_data(args.test_path)
            self.test_loader =DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)
      

    
