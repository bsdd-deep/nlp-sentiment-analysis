# NLP Sentiment Analysis

Advanced sentiment analysis using transformer models and custom neural networks.

## Overview
This project demonstrates sentiment classification on real-world data using state-of-the-art NLP techniques. Multiple approaches are implemented from traditional ML to modern transformers.

## Models
1. **Naive Bayes** - Baseline using TF-IDF
2. **Deep Learning** - Custom LSTM architecture
3. **BERT** - Pre-trained transformer model
4. **DistilBERT** - Lightweight variant

## Datasets
- Amazon Reviews (negative/positive)
- Twitter Sentiment (3-class: negative/neutral/positive)
- Movie Reviews (binary classification)

## Evaluation Results
- Naive Bayes: 0.82 accuracy
- LSTM: 0.87 accuracy
- BERT: 0.93 accuracy
- DistilBERT: 0.91 accuracy

## Key Features
- Text preprocessing & tokenization
- Word embeddings (Word2Vec, FastText)
- Attention visualization
- Inference pipeline
- Model comparison

## Usage
```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(model='bert')
result = analyzer.predict("This product is amazing!")
print(result)  # {'sentiment': 'positive', 'confidence': 0.98}
```

## Architecture
```
Input Text
  ↓
Tokenization + Padding
  ↓
BERT Embedding Layer
  ↓
Attention + Pooling
  ↓
Classification Head
  ↓
Softmax Output
```
# Additional Documentation

## Installation & Setup
see README
