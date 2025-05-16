import streamlit as st
from datetime import datetime
import asyncio
import pytesseract
import pdfplumber
from PIL import Image
import io
import os
import sqlite3
from transformers import BertForTokenClassification, BertTokenizer
from utils import load_tag_map, load_tokenizer  # Assuming these are imported from utils.py
from database import save_invoice_data, create_database

# Streamlit interface starts here..
st.title('Invoice Processing System')
st.sidebar.header('Invoice Operations')
operation = st.sidebar.selectbox('Select Operation', ['Upload Invoice', 'View Saved Data'])

def load_model():
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"  # Adjust model name based on your config
    tokenizer_name = "bert-base-uncased"  # Adjust tokenizer name based on your config
    model = BertForTokenClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

# let us extract text from image using pytesseract OCR method
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return None

# Extract the pdf files
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to extract information from the invoice using Name Entity Recognizer
def extract_invoice_data(invoice_text, model, tokenizer):
    inputs = tokenizer(invoice_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_tags = torch.argmax(logits, dim=-1).squeeze().tolist()
    
    tag_map = load_tag_map()
    tokens = tokenizer.tokenize(invoice_text)
    extracted_data = {}

    for i, tag in enumerate(predicted_tags):
        if tag != 0:  # Ignore 'O' tag (Outside any entity)
            entity = tag_map.get(tag)
            if entity:
                extracted_data[entity] = tokens[i]
    
    return extracted_data

# Take the invoice data processing
async def process_invoice(invoice_file):
    # Extract text from uploaded file (image or PDF)
    if invoice_file.type.startswith("image"):
        invoice_text = extract_text_from_image(invoice_file)
    elif invoice_file.type == "application/pdf":
        invoice_text = extract_text_from_pdf(invoice_file)
    else:
        st.error("Invalid file type. Please upload an image or PDF.")
        return

    if not invoice_text:
        st.error("Failed to extract text from the document. Please try again with a different file.")
        return

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # Extract invoice data using NER
    extracted_data = extract_invoice_data(invoice_text, model, tokenizer)
    
    # Display the extracted data
    st.write(f"Extracted Invoice Data: {extracted_data}")
    logging.info(f"Extracted Data: {extracted_data}")

    # Define the columns and table name (from the database part)
    table_name = 'invoice_data'
    columns = {
        'amount': 'REAL',
        'vendor': 'TEXT',
        'vendor_type': 'INTEGER',
        'date_feature': 'INTEGER',
        'decision': 'TEXT',
        'timestamp': 'DATETIME'
    }

    # Create the database table dynamically if it doesn't exist
    create_database(table_name, columns)

    # Add the extracted data into the document_data dictionary
    document_data = {
        'amount': extracted_data.get('AMOUNT', 0.0),
        'vendor': extracted_data.get('VENDOR', 'Unknown'),
        'vendor_type': 1,  # You can add logic to map vendor types if needed
        'date_feature': 19134,  # You can add logic to convert dates to features
        'decision': 'Pending',  # This is a placeholder decision
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Expected columns:
    expected_columns = list(columns.keys())

    # Saving the extracted data to the database
    await save_invoice_data(table_name, document_data, expected_columns)

# Streamlit app selection:
if operation == 'Upload Invoice':
    # Invoice file upload (image or PDF)
    uploaded_file = st.file_uploader("Upload Invoice File", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image (if it's an image)
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Invoice', use_column_width=True)
        
        # Process the uploaded invoice
        if st.button("Extract and Save Invoice Data"):
            asyncio.run(process_invoice(uploaded_file))

elif operation == 'View Saved Data':
    # Display saved data from the database
    conn = sqlite3.connect('invoice_data.db')
    c = conn.cursor()
    
    # Retrieve all saved data
    c.execute("SELECT * FROM invoice_data")
    rows = c.fetchall()
    
    # Display data in a table format
    if rows:
        st.write("Saved Invoice Data:")
        st.write(rows)
    else:
        st.write("No data found in the database.")
