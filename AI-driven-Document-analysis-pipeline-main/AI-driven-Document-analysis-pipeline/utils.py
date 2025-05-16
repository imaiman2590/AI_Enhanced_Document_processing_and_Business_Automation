# src/utils.py

import torch
from transformers import BertTokenizer

def load_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)

def load_tag_map():
    return {
        0: 'O',               # Outside any entity (default label)
        
        # Banking-specific entities
        1: 'BANK_NAME',       # Bank name (e.g., "Bank of America")
        2: 'ACCOUNT_NUMBER',  # Account number (e.g., "1234567890")
        3: 'CURRENCY',        # Currency type (e.g., "USD", "EUR")
        
        # Supply Chain-specific entities
        4: 'PRODUCT',         # Product name (e.g., "Laptop", "Shoes")
        5: 'SUPPLIER',        # Supplier name (e.g., "ABC Supplies")
        6: 'SHIPMENT_DATE',   # Shipment date (e.g., "2025-06-15")
        
        # Common entities
        7: 'INVOICE_NUMBER',  # Invoice number (e.g., "INV-12345")
        8: 'VENDOR',          # Vendor name (e.g., "XYZ Corp.")
        9: 'AMOUNT',          # Amount mentioned in the document (e.g., "$5000")
        10: 'DATE',            # Date related to the document (e.g., "2025-05-16")
        11: 'EXPIRATION_DATE'  # Expiration date (e.g., "2025-12-31")
    }
