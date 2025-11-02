

# Imports

import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter





# Classes & Functions

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))



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

def encode_sentence(text):
    return [vocab.get(word, vocab['<UNK>']) for word in tokenize(text)]


if __name__ == "__main__":

    # Dataset

    dataset = load_dataset("imdb")
    train = dataset['train']
    test = dataset['test'].select(range(2500))


    print(f"Number of training examples: {len(train)}")

    # Data Preprocessing
    vocab_size = 10000
    counter = Counter()

    for example in train:
        counter.update(tokenize(example['text']))

    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    print(f"Vocabulary size: {len(vocab)}")

    # DataLoader
    train_dataset = IMDBDataset(train)
    test_dataset = IMDBDataset(test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    print("DataLoaders created.")




    #! Initialize model

    model = RNNClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, output_dim=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    print("Model initialized.")

    print(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    print("Starting training...")



    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(texts)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                preds = model(texts)
                predicted = preds.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {correct/total:.2f}")


    print("Training complete.")



    # Saving the model

    torch.save(model.state_dict(), "models/rnn")
    print("Model saved successfully.")


    # Testing the saved model

    model = RNNClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, output_dim=2)
    model.load_state_dict(torch.load("models/rnn"))
    model.eval()

    # Function to predict sentiment
    def predict_sentiment(text):
        encoded = torch.tensor(encode_sentence(text), dtype=torch.long).unsqueeze(0)  # add batch dim
        with torch.no_grad():
            output = model(encoded)
            prediction = torch.argmax(output, dim=1).item()
            return "Positive" if prediction == 1 else "Negative"

    # Test sentence
    test_text = "the movie sucked!"
    print(f"Sentence: '{test_text}'")
    print("Predicted sentiment:", predict_sentiment(test_text))
    print("Model tested successfully.")