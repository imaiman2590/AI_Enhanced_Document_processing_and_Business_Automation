# config.py
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"  
TOKENIZER_NAME = "bert-base-uncased"  

# Network and NER model hyperparameters
VOCAB_SIZE = 10000  
EMBED_DIM = 100  
HIDDEN_DIM = 256  
NUM_TAGS = 12  

DATABASE_PATH = 'invoice_data.db'  

DATABASE_CONFIG = {
    'host': 'localhost',  
    'user': 'root', 
    'password': 'password',  
    'database': 'document_processing'  
}

LOGGING_LEVEL = 'INFO'  # Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
