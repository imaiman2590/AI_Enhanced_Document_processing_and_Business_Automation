# src/utils.py

import torch
from transformers import BertTokenizer

def load_tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)

def load_tag_map():
    return {
        0: 'O',               
        
        # Banking-specific entities
        1: 'BANK_NAME',       
        2: 'ACCOUNT_NUMBER',  
        3: 'CURRENCY',        
        
        # Supply Chain-specific entities
        4: 'PRODUCT',         
        5: 'SUPPLIER',        
        6: 'SHIPMENT_DATE',   
        
        # Common entities
        7: 'INVOICE_NUMBER',  
        8: 'VENDOR',          
        9: 'AMOUNT',          
        10: 'DATE',            
        11: 'EXPIRATION_DATE'  # Expiration date (e.g., "2025-12-31")
    }
