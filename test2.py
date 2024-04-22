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

data = pd.read_csv(TRAIN_PATH)

print(data)

