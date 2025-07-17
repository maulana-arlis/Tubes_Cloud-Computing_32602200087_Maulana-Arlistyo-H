from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
model = AutoModelForSequenceClassification.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")

label_map = model.config.id2label  # otomatis ambil dari metadata model

def analyze_sentiment(text):
    text = text.strip().strip('"').strip("'")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

    return {
        "label": label_map[pred].lower(),  # hasil seperti "positive"
        "confidence": round(confidence, 3)
    }
