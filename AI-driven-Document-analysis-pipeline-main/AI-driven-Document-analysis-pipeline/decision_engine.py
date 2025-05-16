import torch
import pickle
import numpy as np
from datetime import datetime
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.preprocessing import StandardScaler
import os

class DecisionEngineAdvanced:
    def __init__(
        self,
        input_dim=2,
        vendor_to_index_path="model/vendor_to_index.pkl",
        model_path="model/decision_model_advanced.pth",
        scaler_path="model/scaler.pkl",
        vendor_emb_dim=8,
        conv_out_channels=[64, 128],
        attention_heads=4,
        dropout_prob=0.5,
        ner_model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
    ):
        # Load vendor vocab
        with open(vendor_to_index_path, 'rb') as f:
            self.vendor_to_index = pickle.load(f)

        self.vendor_vocab_size = len(self.vendor_to_index)
        
        # Initialize the decision model
        self.model = DecisionNNAdvanced(
            input_dim=input_dim,
            vendor_vocab_size=self.vendor_vocab_size,
            vendor_emb_dim=vendor_emb_dim,
            conv_out_channels=conv_out_channels,
            attention_heads=attention_heads,
            dropout_prob=dropout_prob
        )

        # Load pre-trained model weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        # Load the scaler for numerical features
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load NER model and tokenizer
        self.ner_model = BertForTokenClassification.from_pretrained(ner_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(ner_model_name)
        self.ner_model.eval()

        # Load the NER tag map
        self.tag_map = self.load_tag_map()

    def load_tag_map(self):
        return {
            0: 'O',
            1: 'BANK_NAME',
            2: 'ACCOUNT_NUMBER',
            3: 'CURRENCY',
            4: 'PRODUCT',
            5: 'SUPPLIER',
            6: 'SHIPMENT_DATE',
            7: 'INVOICE_NUMBER',
            8: 'VENDOR',
            9: 'AMOUNT',
            10: 'DATE',
            11: 'EXPIRATION_DATE'
        }

    def extract_features(self, document_data):
        """
        Extract relevant features (including NER tags) for the decision model.
        """
        # Base numerical features (e.g., 'amount' and 'date')
        base_features = [
            document_data.get('amount', 0),
            self.date_feature(document_data.get('date', ''))
        ]
        base_features = np.array(base_features).reshape(1, -1)
        
        # Apply scaling to numerical features
        scaled = self.scaler.transform(base_features)
        numerical_tensor = torch.tensor(scaled, dtype=torch.float32)

        # Vendor index (with fallback for unknown vendors)
        vendor_name = document_data.get('vendor', '').strip()
        vendor_idx = self.vendor_to_index.get(vendor_name, self.vendor_to_index.get('__UNK__'))
        vendor_tensor = torch.tensor([vendor_idx], dtype=torch.long)

        # Extract NER features
        ner_tags = self.extract_ner_tags(document_data.get('text', ''))

        # Append the NER tag information to the feature set
        # Optionally: map NER tags to numerical representations or include them as categorical inputs
        ner_features = self.ner_tags_to_features(ner_tags)
        numerical_tensor = torch.cat([numerical_tensor, torch.tensor(ner_features, dtype=torch.float32)], dim=1)

        return numerical_tensor, vendor_tensor

    def extract_ner_tags(self, text):
        """
        Extract NER tags from the document text using the pre-trained NER model.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
        
        # Map the predicted token indices to entity labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        ner_tags = [self.tag_map[pred.item()] for pred in predictions[0]]
        
        # Filter the tokens and their corresponding NER tags
        entity_tags = [(token, tag) for token, tag in zip(tokens, ner_tags) if tag != 'O']
        
        return entity_tags

    def ner_tags_to_features(self, ner_tags):
        """
        Convert NER tags into features (e.g., binary features for each entity type).
        """
        # Initialize a feature vector for the entities
        ner_feature_vector = [0] * len(self.tag_map)
        
        # Mark the presence of relevant NER tags
        for _, tag in ner_tags:
            if tag in self.tag_map.values():
                ner_feature_vector[list(self.tag_map.values()).index(tag)] = 1
        
        return ner_feature_vector

    def date_feature(self, date_string):
        """
        Convert a date string to a numerical feature (days since epoch).
        """
        if not date_string:
            return 0
        try:
            date = datetime.strptime(date_string, "%Y-%m-%d")
            return (date - datetime(1970, 1, 1)).days  # Days since epoch
        except Exception:
            return 0

    def make_decision(self, document_data):
        """
        Make a decision based on the document data using the trained model.
        """
        numerical_input, vendor_input = self.extract_features(document_data)

        # Perform forward pass and get output
        with torch.no_grad():
            output = self.model(numerical_input, vendor_input)

        # Get the class with the highest probability (decision)
        decision_idx = torch.argmax(output, dim=1).item()
        
        # Map decision index to the corresponding label
        decision_map = {
            0: 'Approved',
            1: 'Under Review',
            2: 'Flagged for Fraud',
            3: 'Rejected'
        }
        
        # Include NER extracted information in decision output for transparency
        ner_info = document_data.get('text', '')
        ner_tags = self.extract_ner_tags(ner_info)
        
        decision = decision_map[decision_idx]
        return {
            'decision': decision,
            'ner_tags': ner_tags
        }
