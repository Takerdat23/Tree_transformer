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
import py_vncorenlp

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



def createNLIdataset(path): 
    with open(path , 'r') as json_file:
        json_list = list(json_file) 
    dataset = []


    for json_str in json_list:
        instance1 = {}
        instance2 = {}

        result = json.loads(json_str)
        hypo1 = result['sentence1']
        hypo2 = result['sentence2']

        context = result['context']
        label = result['gold_label']

        label_vector = [0, 0 , 0 ,0]
        if label == "entailment": 
            label_vector[0] = 1
        elif label == "contradiction": 
            label_vector[1] = 1 
        elif label == "neutral": 
            label_vector[2] = 1 
        elif label == "other": 
            label_vector[3] = 1 

        instance1['premise'] = context
        instance1['hypo'] = hypo1
        instance1['label']= label_vector

        

        dataset.append(instance1)


        instance2['premise'] = context
        instance2['hypo'] = hypo2
        instance2['label']= label_vector
        dataset.append(instance2)
    return dataset


def getTokenizerData(path): 
    with open(path , 'r') as json_file:
        json_list = list(json_file) 
    dataset = []


    for json_str in json_list:
        instance = {}

        result = json.loads(json_str)
        hypo1 = result['sentence1']
        hypo2 = result['sentence2']

        context = result['context']

        text1 = hypo1 + " " + context

        dataset.append(text1)

        text2 = hypo2 + " " + context

        dataset.append(text2)
    

    return dataset





def cc(arr, no_cuda=False):
    if no_cuda:
        return torch.from_numpy(np.array(arr))
    else:
        return torch.from_numpy(np.array(arr)).cuda()






class NLIDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        hypos = [example["hypo"] for example in batch]
        premises = [example["premise"] for example in batch]
        labels = [example["label"] for example in batch]


        encoded_batch = [self.tokenizer.encode(hypo + " " + premise) for hypo, premise in zip(hypos  , premises)]

        # Get the maximum sequence length
        max_length = 128

        # Pad and truncate the sequences
        for encoded in encoded_batch:
            encoded.pad(max_length)
            encoded.truncate(max_length)

        # Convert the sequences to numpy arrays
        input_ids = torch.tensor([encoded.ids for encoded in encoded_batch])
        attention_mask = torch.tensor([encoded.attention_mask for encoded in encoded_batch])

 
     
        labels_tensor = torch.stack(labels)
       

      
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels_tensor}


class data_utils():
    def __init__(self, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        self.no_cuda = args.no_cuda
        self.train_path = args.train_path

        text_data = getTokenizerData(args.train_path)

        if os.path.exists(os.path.join(args.model_dir,"vocab.json" )) and os.path.exists(os.path.join(args.model_dir,"merges.txt" )): 
            # self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.tokenizer = ByteLevelBPETokenizer.from_file( os.path.join(args.model_dir,"vocab.json" ), os.path.join(args.model_dir,"merges.txt" ))
        else: 
            print("No Tokenizer found")
            # self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            
            tokenizer = ByteLevelBPETokenizer()

            tokenizer.train_from_iterator(text_data, vocab_size=30000, min_frequency=2,
                              special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.save_model(args.model_dir)
            self.tokenizer = ByteLevelBPETokenizer.from_file( os.path.join(args.model_dir,"vocab.json" ), \
                                                             os.path.join(args.model_dir,"merges.txt" ))

          
        data_collator = NLIDataCollator(self.tokenizer)
       
        dataset = createNLIdataset(args.train_path)
        val_dataset =  createNLIdataset(args.valid_path)
        
        self.train_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
        self.val_loader =DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)


        if args.test : 

          
            test_dataset = createNLIdataset(args.test_path)
            self.test_loader =DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)
      

    
