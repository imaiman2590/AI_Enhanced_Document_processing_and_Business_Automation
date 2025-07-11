# src/ner_inference.py

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


class NEREngine:
    def __init__(self, model_name="dslim/bert-base-NER"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",  # Combine B/I tags
            device=0 if torch.cuda.is_available() else -1
        )

    def extract_entities(self, text):
        results = self.ner_pipeline(text)
        return [{"text": r["word"], "type": r["entity_group"]} for r in results]

    def extract_entities_batch(self, texts):
        return [self.extract_entities(text) for text in texts]

    def test_inference_speed(self, texts):
        import time
        start = time.time()
        _ = self.extract_entities_batch(texts)
        end = time.time()
        print(f"Inference time for {len(texts)} texts: {end - start:.4f} seconds")
