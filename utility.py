import PyPDF2
from transformers import BertTokenizer, BertModel
import torch

class Utility:
    def extract_text_from_pdf(pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            for page_num in range(reader.numPages):
                text += reader.getPage(page_num).extractText()
        return text
    
    def generate_vectors(text):
        model_name = "bert-base-uncased"  # Example model name
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state
        return embeddings
    




