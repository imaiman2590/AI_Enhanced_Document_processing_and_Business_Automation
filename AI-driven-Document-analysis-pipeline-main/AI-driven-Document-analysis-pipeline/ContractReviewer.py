import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class ContractReviewer:
    def __init__(self, model_path='t5-base', tokenizer_path='t5-base', device=None):
        """
        Initializes the ContractReviewer with a pretrained T5 model and tokenizer.
        Args:
            model_path (str): Path or HuggingFace model ID for the T5 model.
            tokenizer_path (str): Path or HuggingFace model ID for the tokenizer.
            device (str): 'cpu' or 'cuda'. Auto-detects if not provided.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

    def preprocess(self, contract_text):
        """
        Prepares the contract text for the T5 model.
        """
        return self.tokenizer(
            contract_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

    def extract_clauses(self, contract_text, max_length=150):
        """
        Extracts key clauses or summary from a contract.
        Args:
            contract_text (str): Raw contract input text.
            max_length (int): Maximum length of the generated output.
        Returns:
            str: Generated clause summary or key terms.
        """
        inputs = self.preprocess(contract_text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def extract_batch(self, contract_texts, max_length=150):
        """
        Processes a batch of contract texts and returns summaries for each.
        Args:
            contract_texts (List[str]): List of contract texts.
            max_length (int): Max length of output summary.
        Returns:
            List[str]: List of extracted clause summaries.
        """
        results = []
        for text in contract_texts:
            try:
                summary = self.extract_clauses(text, max_length=max_length)
                results.append(summary)
            except Exception as e:
                results.append("Error processing contract")
        return results
