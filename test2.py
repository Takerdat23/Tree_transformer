import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer, AutoTokenizer
import torch
import numpy as np 
from utils import * 
from torch.utils.data import Dataset, DataLoader
from models import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/phobert-base")
model_en2vi = AutoModel.from_pretrained("vinai/phobert-base")
text = " Xin chào bạn"
input = tokenizer_en2vi(text ,ma)
 