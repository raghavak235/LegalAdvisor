from PyPDF2 import PdfReader
from transformers import BertTokenizer, BertModel
import torch
import fitz

class Utility:
    def extract_text_from_pdf(pdf_path):
        # Open the PDF file
        with open(pdf_path, "rb") as file:
            # Create a PdfReader object
            pdf_reader = PdfReader(file)
            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
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
    
    def extract_text_from_pdf_using_fitz(pdf_path):
        pdf_document = fitz.open(pdf_path)
        # Extract text from each page
        text = ''
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

    




