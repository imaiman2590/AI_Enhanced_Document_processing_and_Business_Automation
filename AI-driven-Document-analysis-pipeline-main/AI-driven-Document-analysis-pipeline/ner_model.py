# src/ner_model.py

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
            aggregation_strategy="simple",  # Aggregate B/I labels into full entities
            device=0 if torch.cuda.is_available() else -1
        )

    def predict_tags(self, tokens):
        text = " ".join(tokens)
        results = self.ner_pipeline(text)
        return [{"text": r["word"], "type": r["entity_group"]} for r in results]

    def predict_tags_batch(self, list_of_token_lists):
        return [self.predict_tags(tokens) for tokens in list_of_token_lists]
