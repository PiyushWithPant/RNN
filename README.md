# 🎬 IMDB Movie Review Sentiment Analysis (RNN with PyTorch)

This repository contains an implementation of a Recurrent Neural Network (RNN) for text generation.

---

## Overview 📝

This project implements a **Recurrent Neural Network (RNN)** using **PyTorch** to classify the sentiment (Positive/Negative) of movie reviews from the classic **IMDB dataset**.

The model is trained using word embeddings and a simple RNN layer and achieves a good baseline performance on the sentiment classification task.

---

## Warning ⚠️

**Model Not Optimized** - This is an experimental implementation and the model may not produce optimal results. Use with caution.

---

## Features ⭐

- Dataset: IMDB reviews from `datasets` library
- Model: Custom RNN with embedding and linear layer
- Training and evaluation pipeline
- Custom tokenization and vocabulary building
- Predict sentiment for any review after training
- Easy model saving and loading

---

## Requirements 🛠️

- Python 3.12+

- `requirements.txt` - Use this file to install all the required libraries.

---

## Getting Started 🚀

1. Clone this repository
2. Install dependencies
3. Run training scripts located at - `src/`
4. Generate text using the trained model

---

## What are the Scripts?

1. `rnn.py` : The main RNN initialization & training.
2. `embedding.ipynb` & `embeddings_method2.ipynb` : Embedding tutorial to understand how to get word embeddings from text.
3. `eval.ipynb` : Evaluation of the RNN model
4. `app.py` : Streamlit Application

---

## License 📄

MIT License

> By Piyush Pant (पीयूष पंत)
