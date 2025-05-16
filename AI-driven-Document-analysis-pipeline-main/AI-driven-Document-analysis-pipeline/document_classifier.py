import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import asyncio

class DocumentClassifier:
    def __init__(self, model_path, tokenizer_path='roberta-base', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RobertaForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

    def preprocess_text(self, text):
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    def classify(self, text):
        try:
            inputs = self.preprocess_text(text)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
            return predicted_class
        except Exception:
            return None

    async def classify_batch(self, texts):
        predictions = []
        tasks = [self._classify_single_async(text, predictions) for text in texts]
        await asyncio.gather(*tasks)
        return predictions

    async def _classify_single_async(self, text, predictions):
        result = self.classify(text)
        predictions.append(result)

    def evaluate(self, dataloader, metric="accuracy"):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = labels.to(self.device)

                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_classes = torch.argmax(logits, dim=-1)

                correct_predictions += (predicted_classes == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        return accuracy

    def cache_model(self, model_path):
        if not hasattr(self, 'cached_model') or self.cached_model != model_path:
            self.model = RobertaForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.cached_model = model_path

# Example usage
if __name__ == "__main__":
    model_path = "path_to_classification_model.pth"
    classifier = DocumentClassifier(model_path=model_path)

    # Single document classification
    predicted_class = classifier.classify("This is an invoice document")
    print(f"Predicted class: {predicted_class}")

    # Batch document classification
    async def batch_classification():
        documents = ["Invoice 1", "Purchase order document", "Invoice for vendor XYZ"]
        predictions = await classifier.classify_batch(documents)
        print(f"Batch predictions: {predictions}")

    asyncio.run(batch_classification())
