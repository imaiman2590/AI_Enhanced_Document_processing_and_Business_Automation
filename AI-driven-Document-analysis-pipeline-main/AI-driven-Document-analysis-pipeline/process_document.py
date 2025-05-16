# src/process_document.py

from src.extract_text import extract_text_from_pdf, extract_text_from_image, extract_text_from_csv, extract_text_from_xlsx
from src.extract_entities import extract_entities, load_ner_model
from src.decision_engine import DecisionEngineAdvanced
from src.database import save_invoice_data
from src.utils import load_tokenizer, load_tag_map

def process_document(file_path, file_type, decision_engine, ner_model=None, tokenizer=None, tag_map=None):
    # Extract text based on document type
    file_extractors = {
        "pdf": extract_text_from_pdf,
        "image": extract_text_from_image,
        "csv": extract_text_from_csv,
        "xlsx": extract_text_from_xlsx
    }

    if file_type not in file_extractors:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Extract the text from the document using the appropriate extractor
    text = file_extractors[file_type](file_path)
    
    # Initialize document data
    document_data = {}

    # If NER model, tokenizer, and tag map are provided, extract entities
    if ner_model and tokenizer and tag_map:
        entities = extract_entities(text, ner_model, tokenizer, tag_map)
        document_data.update(entities)
    
    # Make decision using the decision engine
    decision = decision_engine.make_decision(document_data)
    
    # Add decision to the document data
    document_data["decision"] = decision
    
    # Save document data to the database
    save_invoice_data(document_data)
    
    return document_data
