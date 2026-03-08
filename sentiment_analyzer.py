import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model='distilbert-base-uncased-finetuned-sst-2-english'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text, return_all_scores=False):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        scores = torch.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(scores).item()
        
        labels = ['negative', 'positive']
        result = {
            'sentiment': labels[predicted_class],
            'confidence': scores[predicted_class].item()
        }
        
        if return_all_scores:
            result['scores'] = {labels[i]: scores[i].item() for i in range(len(labels))}
        
        return result
    
    def batch_predict(self, texts):
        return [self.predict(text) for text in texts]

# Integration example
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I absolutely love this product!",
        "The worst experience ever",
        "It's okay, nothing special"
    ]
    
    for text in test_texts:
        result = analyzer.predict(text, return_all_scores=True)
        print(f"Text: {text}")
        print(f"Result: {result}\n")
