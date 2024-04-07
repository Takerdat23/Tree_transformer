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




def process_NLI(df, num_labels):
   
    dataset = []
   
    for _, row in df.iterrows():
        data_dict = {}
 
        data_dict["Premise"] = row['Premise']
        
        # Convert sentiment and topic to one-hot encoded tensors
        label =row['label']
       
        oneHot_label = torch.zeros(num_labels)
        if label == "Entailment": 
            oneHot_label[0] = 1
        elif label == "Contradiction" : 
            oneHot_label[1] = 1 
        elif label == "Neutral": 
            oneHot_label[2] = 1 
        else: 
            oneHot_label[3] = 1
        
     
        
     
    
        data_dict["Label"] =   oneHot_label
        dataset.append(data_dict)

    return dataset






def cc(arr, no_cuda=False):
    if no_cuda:
        return torch.from_numpy(np.array(arr))
    else:
        return torch.from_numpy(np.array(arr)).cuda()






class SentimentDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        inputs = [example["comment"] for example in batch]
        labels = [example["label"] for example in batch]


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

        df_train = pd.read_csv(args.train_path,  encoding = 'utf8') 
  
        df_val = pd.read_csv(args.valid_path,  encoding = 'utf8')
        
        if os.path.exists(os.path.join(args.model_dir,"vocab.json" )) and os.path.exists(os.path.join(args.model_dir,"merges.txt" )): 
            # self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.tokenizer = ByteLevelBPETokenizer.from_file( os.path.join(args.model_dir,"vocab.json" ), os.path.join(args.model_dir,"merges.txt" ))
        else: 
            print("No Tokenizer found")
            # self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            
            tokenizer = ByteLevelBPETokenizer()

            tokenizer.train_from_iterator(df_train["comment"], vocab_size=30000, min_frequency=2,
                              special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            tokenizer.save_model(args.model_dir)
            self.tokenizer = ByteLevelBPETokenizer.from_file( os.path.join(args.model_dir,"vocab.json" ), \
                                                             os.path.join(args.model_dir,"merges.txt" ))

          
        data_collator = SentimentDataCollator(self.tokenizer)
       
        dataset = process_NLI(df_train)
        val_dataset =  process_NLI(df_val)
        
        self.train_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
        self.val_loader =DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)


        if args.test : 

            df_test = pd.read_csv(args.test_path,  encoding = 'utf8')
        
            test_dataset = process_NLI(df_test)
            self.test_loader =DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collator)
      

    
