import torch
import re
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm


# Classes & Functions

class IMDBDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoded = torch.tensor(encode_sentence(text), dtype=torch.long)
        return encoded, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, padding_value=vocab['<PAD>'], batch_first=True)
    return texts_padded, torch.tensor(labels)

def tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().split()

def encode_sentence(text, vocab):
    return [vocab.get(word, vocab['<UNK>']) for word in tokenize(text)]



# Dataset

dataset = load_dataset("imdb")
train = dataset['train']
test = dataset['test']

# Data Preprocessing
vocab_size = 10000
counter = Counter()

for example in train:
    counter.update(tokenize(example['text']))

# Keep most common words
vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

# DataLoader
train_dataset = IMDBDataset(train)
test_dataset = IMDBDataset(test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)






