import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
import torch
import numpy as np 
from utils import * 
from torch.utils.data import Dataset, DataLoader
from models import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

TRAIN_PATH = './data/UIT-VSMEC/Train.csv'
VAL_PATH = './data/UIT-VSMEC/Dev.csv'
TEST_PATH = './data/UIT-VSMEC/Test.csv'





def getDataset(path): 
    dataset = []
    data = pd.read_csv(path, encoding = 'utf-8')

    labels = pd.unique(data['Emotion'])
    
    for i , row in data.iterrows(): 
        instance = {}

        instance["comment"] = row['Sentence']
        label = row['Emotion']
        label_vector = torch.zeros(7)
        if label == 'Disgust': 
            label_vector[0] = 1 
        elif label == 'Enjoyment': 
            label_vector[1]= 1 
        elif label == 'Anger': 
            label_vector[2]= 1
        elif label == 'Surprise': 
            label_vector[3]= 1 
        elif label == 'Sadness': 
            label_vector[4]= 1 
        elif label == 'Fear': 
            label_vector[5]= 1 
        elif label == 'Other': 
            label_vector[6]= 1 
        instance['label'] = label_vector

        dataset.append(instance)
    
    return dataset

dataset = getDataset(TRAIN_PATH)
print(len(dataset))






