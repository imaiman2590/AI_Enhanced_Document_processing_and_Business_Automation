# src/text_extractor.py

import os
import pdfplumber
import pytesseract
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class TextExtractorEngine:
    def __init__(self, enable_parallel=True, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
        """
        Initialize the TextExtractorEngine with an optional transformer model for NER.

        Args:
            enable_parallel (bool): Whether to enable parallel processing for batch extraction.
            model_name (str): The name of the pre-trained transformer model to use for NER.
        """
        self.enable_parallel = enable_parallel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def extract(self, file_path: str) -> Union[str, None]:
        """
        Extract text from a file based on its extension and process it using a transformer model.

        Args:
            file_path (str): The path to the file to extract text from.

        Returns:
            str or None: Extracted and processed text or None if extraction fails.
        """
        ext = os.path.splitext(file_path)[1].lower()

        try:
            match ext:
                case ".pdf":
                    text = self._extract_pdf(file_path)
                    return self._process_text_with_transformer(text)
                case ".csv":
                    return self._extract_csv(file_path)
                case ".xlsx" | ".xls":
                    return self._extract_xlsx(file_path)
                case ".jpg" | ".jpeg" | ".png":
                    return self._extract_image(file_path)
                case _:
                    # Unsupported file type
                    return None
        except Exception as e:
            # Handling errors during extraction with specific error message
            print(f"Error extracting text from {file_path}: {e}")
            return None

    def extract_batch(self, file_paths: list[str]) -> Dict[str, Union[str, None]]:
        """
        Extract text from multiple files, optionally in parallel.
        
        Args:
            file_paths (list): List of file paths to extract text from.
        
        Returns:
            dict: A dictionary where the keys are file paths and the values are extracted text or None.
        """
        results = {}
        if self.enable_parallel:
            with ThreadPoolExecutor() as executor:
                future_results = executor.map(self.extract, file_paths)
                results = dict(zip(file_paths, future_results))
        else:
            for file in file_paths:
                results[file] = self.extract(file)
        return results

    # ------------------- Individual File Type Handlers ------------------------

    def _extract_pdf(self, path: str) -> str:
        """
        Extract text from PDF files using pdfplumber.
        
        Args:
            path (str): The path to the PDF file.
        
        Returns:
            str: Extracted text from the PDF.
        """
        text = ''
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
        return text.strip()

    def _extract_csv(self, path: str) -> str:
        """
        Extract text from CSV files by reading and formatting the data.
        
        Args:
            path (str): The path to the CSV file.
        
        Returns:
            str: Extracted text from the CSV.
        """
        try:
            df = pd.read_csv(path, encoding='utf-8', engine='python').fillna('')
            return df.to_string(index=False, header=False)
        except Exception as e:
            print(f"Error extracting text from CSV: {e}")
            return ""

    def _extract_xlsx(self, path: str) -> str:
        """
        Extract text from XLSX files by reading sheet data.
        
        Args:
            path (str): The path to the XLSX file.
        
        Returns:
            str: Extracted text from the XLSX.
        """
        text = ''
        try:
            sheets = pd.read_excel(path, sheet_name=None)
            for name, sheet in sheets.items():
                text += f"\nSheet: {name}\n"
                text += sheet.fillna('').to_string(index=False, header=False)
        except Exception as e:
            print(f"Error extracting text from XLSX: {e}")
        return text.strip()

    def _extract_image(self, path: str) -> str:
        """
        Extract text from image files using OCR (Tesseract).
        
        Args:
            path (str): The path to the image file.
        
        Returns:
            str: Extracted text from the image.
        """
        try:
            image = Image.open(path)
            processed_image = self._preprocess_image(image)
            return pytesseract.image_to_string(processed_image).strip()
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for OCR (Tesseract) by converting to grayscale,
        applying a threshold, enhancing sharpness, and filtering.
        
        Args:
            image (Image.Image): The image to preprocess.
        
        Returns:
            Image.Image: The processed image ready for OCR.
        """
        gray = image.convert('L')
        threshold = gray.point(lambda p: p > 200 and 255)  # Simple thresholding
        sharp = ImageEnhance.Sharpness(threshold).enhance(2.0)  # Enhancing sharpness
        final = sharp.filter(ImageFilter.MedianFilter(size=3))  # Applying median filter
        return final

    def _process_text_with_transformer(self, text: str) -> str:
        """
        Process the extracted text using a transformer model for NER or document understanding.
        
        Args:
            text (str): The extracted text from a document.
        
        Returns:
            str: Processed text with named entities or further analysis.
        """
        # Tokenizing and passing text through the transformer model
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        
        # Get predictions and map them to labels
        predictions = torch.argmax(outputs, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predicted_labels = predictions[0].tolist()

        # Combine tokens and predicted labels
        entity_results = []
        for token, label in zip(tokens, predicted_labels):
            label_name = self.model.config.id2label[label]
            if label_name != 'O':  # If it's not the "O" label, it's an entity
                entity_results.append((token, label_name))
        
        # Return the extracted entities in a readable format
        return "\n".join([f"{token}: {label}" for token, label in entity_results])
