import sqlite3
from datetime import datetime
import asyncio
import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
import logging
import pickle
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load NER model and tokenizer
def load_ner_model(model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    nlp_ner = pipeline("ner", model=model_name, tokenizer=model_name)
    return nlp_ner

def extract_entities_from_text(text, nlp_ner):
    entities = nlp_ner(text)
    entity_dict = {entity['label']: entity['word'] for entity in entities}
    return entity_dict

# Function to extract features from document data, now using NER
async def save_invoice_data_with_ner(table_name, document_data, expected_columns, nlp_ner):
    try:
        # Validate the input data against expected columns
        validate_document_data(document_data, expected_columns)

        # Extract entities from the document text using NER
        document_text = document_data.get('text', '')  # Assuming the document text is available
        extracted_entities = extract_entities_from_text(document_text, nlp_ner)

        # Use the extracted entities to populate document data
        document_data['bank_name'] = extracted_entities.get('BANK_NAME', '')
        document_data['account_number'] = extracted_entities.get('ACCOUNT_NUMBER', '')
        document_data['currency'] = extracted_entities.get('CURRENCY', '')
        document_data['vendor'] = extracted_entities.get('VENDOR', '')
        document_data['amount'] = extracted_entities.get('AMOUNT', '')

        # Prepare the columns and values for insertion
        columns = ", ".join(document_data.keys())
        values = tuple(document_data.values())

        # Insert data into the database dynamically
        conn = sqlite3.connect('invoice_data.db')
        c = conn.cursor()

        placeholders = ", ".join(["?" for _ in document_data])
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        c.execute(insert_query, values)
        conn.commit()

        logging.info(f"Invoice data saved to '{table_name}': {document_data}")
        print(f"Saved document data: {document_data}")

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        print(f"Error: {ve}")
    
    except sqlite3.DatabaseError as db_err:
        logging.error(f"Database error: {db_err}")
        print(f"Error: {db_err}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
    finally:
        conn.close()

async def main():
    # Example: Dynamic table name and columns
    table_name = 'invoice_data'

    # Columns and their data types (can change for different datasets)
    columns = {
        'amount': 'REAL',
        'vendor': 'TEXT',
        'vendor_type': 'INTEGER',
        'date_feature': 'INTEGER',
        'decision': 'TEXT',
        'timestamp': 'DATETIME',
        'bank_name': 'TEXT',
        'account_number': 'TEXT',
        'currency': 'TEXT'
    }

    # table creation
    create_database(table_name, columns)

    # Collected data's will be like below:
    document_data = {
        'amount': 1500.75,
        'vendor': 'Amazon',
        'vendor_type': 1,
        'date_feature': 1913, 
        'decision': 'Approved',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'text': "The invoice was paid through Bank of America, Account 1234567890, USD. The vendor is Amazon."
    }

    # List of expected columns for validation
    expected_columns = list(columns.keys())

    # Load NER model and tokenizer
    nlp_ner = load_ner_model()

    # Save the document data to the database with NER processing
    await save_invoice_data_with_ner(table_name, document_data, expected_columns, nlp_ner)

# Run the async save operation
asyncio.run(main())
