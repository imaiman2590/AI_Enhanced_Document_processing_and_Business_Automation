# src/ner_inference.py

import torch
import pickle
from src.ner_model import BiLSTM_CRF_Attn
from src.utils import tokenize
import time


class NEREngine:
    def __init__(self, model_path="model/ner_model.pth", config_path="model/ner_config.pkl"):
        # Load model and configuration
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        with open(config_path, "rb") as f:
            config = pickle.load(f)

        # Using the updated load_tag_map function
        self.word_to_index = config["word_to_index"]
        self.tag_to_index = config["tag_to_index"]  # Updated tag map for banking and supply chain
        self.index_to_tag = {v: k for k, v in self.tag_to_index.items()}
        self.tokenizer = config.get("tokenizer", lambda x: x.split())  # This is a fallback tokenizer
        self.max_length = config.get("max_length", 128)

        # Initialize the model with the new number of tags
        self.model = BiLSTM_CRF_Attn(
            vocab_size=len(self.word_to_index),
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_tags=len(self.tag_to_index),
            embedding_matrix=config.get("embedding_matrix", None)
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _tokens_to_ids(self, tokens):
        unk = self.word_to_index.get("__UNK__", 0)
        return [self.word_to_index.get(token, unk) for token in tokens]

    def preprocess(self, text):
        # Tokenize the input text
        tokens = tokenize(text)  # Assuming this function handles the necessary tokenization
        token_ids = self._tokens_to_ids(tokens)
        return tokens, torch.tensor([token_ids], dtype=torch.long).to(self.device)

    def extract_entities(self, text):
        # Extract entities for a single text
        tokens, input_tensor = self.preprocess(text)
        with torch.no_grad():
            emissions, _, _ = self.model(input_tensor)
            tag_ids = self.model.crf.decode(emissions)[0]
        return self._decode_entities(tokens, tag_ids)

    def extract_entities_batch(self, texts):
        # Extract entities for a batch of texts
        batch_tokens = [tokenize(text) for text in texts]
        batch_ids = [self._tokens_to_ids(tokens) for tokens in batch_tokens]
        max_len = max(len(ids) for ids in batch_ids)
        padded = [ids + [0] * (max_len - len(ids)) for ids in batch_ids]

        input_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)

        with torch.no_grad():
            emissions, _, _ = self.model(input_tensor)
            all_tag_ids = self.model.crf.decode(emissions)

        return [self._decode_entities(tokens, tag_ids)
                for tokens, tag_ids in zip(batch_tokens, all_tag_ids)]

    def _decode_entities(self, tokens, tag_ids):
        # Decode the tags to actual entities
        entities = []
        current_entity = None

        for token, tag_id in zip(tokens, tag_ids):
            tag = self.index_to_tag.get(tag_id, "O")  # Default to "O" if no tag found
            if tag.startswith("B-"):  # Beginning of an entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"text": token, "type": tag[2:]}  # Remove 'B-' to get the entity type
            elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["type"]:  # Inside an entity
                current_entity["text"] += f" {token}"
            else:  # If tag is outside the current entity, close the previous entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Add the last entity if it's there
        if current_entity:
            entities.append(current_entity)

        return entities

    def test_inference_speed(self, texts):
        # Test the inference speed for a batch of texts
        start = time.time()
        _ = self.extract_entities_batch(texts)
        end = time.time()
        print(f"Inference time for {len(texts)} texts: {end - start:.4f} seconds")
