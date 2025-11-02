import torch
import torch.nn as nn
import re
import streamlit as st
from collections import Counter
from datasets import load_dataset

# -----------------
# Load Vocabulary
# -----------------
dataset = load_dataset("imdb")
train = dataset['train']
vocab_size = 10000
counter = Counter()

def tokenize(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().split()

for example in train:
    counter.update(tokenize(example['text']))

vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def encode_sentence(text):
    return [vocab.get(word, vocab['<UNK>']) for word in tokenize(text)]


# -----------------
# Define Model
# -----------------
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Load trained model
model = RNNClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, output_dim=2)
model.load_state_dict(torch.load("models/rnn"))
model.eval()

def predict_sentiment(text):
    encoded = torch.tensor(encode_sentence(text), dtype=torch.long).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        output = model(encoded)
        prediction = torch.argmax(output, dim=1).item()
        return "Positive 😊" if prediction == 1 else "Negative 😞"


# -----------------
# Streamlit UI
# -----------------



st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="centered")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and find out if the sentiment is **positive** or **negative**!")

review = st.text_area("📝 Enter a movie review", "", height=150)

if st.button("Analyze Sentiment"):
    if review.strip():
        with st.spinner("Analyzing..."):
            sentiment = predict_sentiment(review)
        st.success(f"**Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")
