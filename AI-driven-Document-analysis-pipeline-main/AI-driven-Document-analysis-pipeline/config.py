# config.py

# Path for pre-trained NER model and tokenizer models
MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Replace with the actual model path or name
TOKENIZER_NAME = "bert-base-uncased"  # Tokenizer for BERT (or any other tokenizer)

# Network and NER model hyperparameters
VOCAB_SIZE = 10000  
EMBED_DIM = 100  
HIDDEN_DIM = 256  
NUM_TAGS = 12  

DATABASE_PATH = 'invoice_data.db'  # Path to your SQLite database file

DATABASE_CONFIG = {
    'host': 'localhost',  # pls provide host name
    'user': 'root',  # pls provide usr name
    'password': 'password',  # Pls provide paswd
    'database': 'document_processing'  # pls provide the database name I don't have any
}

LOGGING_LEVEL = 'INFO'  # Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
