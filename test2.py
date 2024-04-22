import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
import torch
import numpy as np 
import json
from utils import * 
from torch.utils.data import Dataset, DataLoader
from models import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TRAIN_PATH = './data/VNLI/Train.jsonl'
VAL_PATH = './data/VNLI/Dev.jsonl'
TEST_PATH = './data/VNLI/test.jsonl'




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

        instance1['premise'] = context
        instance1['hypo'] = hypo1
        instance1['label']= label

        dataset.append(instance1)


        instance2['premise'] = context
        instance2['hypo'] = hypo2
        instance2['label']= label
        dataset.append(instance2)
    return dataset

train = createNLIdataset(TRAIN_PATH)
print(len(train))

        
        
    
    



