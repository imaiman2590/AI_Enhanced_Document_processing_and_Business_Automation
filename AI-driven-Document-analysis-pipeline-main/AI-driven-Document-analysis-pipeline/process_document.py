# src/process_document.py

from src.ner_inference import NEREngine
from src.text_extractor_engine import TextExtractorEngine  # Ensure this is the correct path
from src.decision_engine import DecisionEngineAdvanced
from src.database import save_invoice_data

# Initialize engines once
text_extractor = TextExtractorEngine()
ner_engine = NEREngine()

def process_document(file_path, file_type, decision_engine):
    """
    Extracts text from a document, applies NER, makes a decision, and saves data.

    Args:
        file_path (str): Path to the input document.
        file_type (str): Type of the file (pdf or image).
        decision_engine (DecisionEngineAdvanced): Engine to make decisions from extracted entities.

    Returns:
        dict: Final processed data including extracted entities and decision.
    """
    # Step 1: Extract text using OCR or document model
    if file_type == "pdf":
        text = text_extractor.extract_from_pdf(file_path)
    elif file_type == "image":
        text = text_extractor.extract_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    document_data = {}

    # Step 2: Extract entities using NER
    entities = ner_engine.extract_entities(text)
    for entity in entities:
        entity_type = entity["type"]
        entity_text = entity["text"]

        # Handle multiple values for the same entity type
        if entity_type in document_data:
            if isinstance(document_data[entity_type], list):
                document_data[entity_type].append(entity_text)
            else:
                document_data[entity_type] = [document_data[entity_type], entity_text]
        else:
            document_data[entity_type] = entity_text

    # Step 3: Make a decision
    decision = decision_engine.make_decision(document_data)
    document_data["decision"] = decision

    # Step 4: Save the results
    save_invoice_data(document_data)

    return document_data
