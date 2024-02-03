from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from gru_model import GRUModel
from utility import Utility
import torch
import re

# custom_model.py
class CustomLanguageModel:
    def generate_response(self, prompt):
        # Replace 'example.pdf' with the path to your PDF file
        text = Utility.extract_text_from_pdf('/Users/yaswanthkalvala/gitprojects/LLM/LegalAdvisor/test.pdf')
        #embeddings = Utility.generate_vectors(text)
        #print("Embeddings shape:", embeddings.shape)
        # Tokenize the text
        # This is a simple example using whitespace tokenization and removing punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenization with lowercase conversion and punctuation removal

        # Build vocabulary from tokens
        vocab = {'<pad>': 0, '<unk>': 1}  # Start with special tokens
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)  # Assign a numerical index to each unique token

        # Now you have the vocabulary mapped from tokens extracted from the PDF
        print("Vocabulary:", vocab)
        
        # Convert tokens to indexed tokens using the vocabulary
        indexed_tokens = [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]

        print(indexed_tokens)

        # Convert indexed tokens to PyTorch tensor
        input_tensor = torch.tensor(indexed_tokens).unsqueeze(0)  # Add batch dimension

        print(input_tensor)
        # Parameters
        # Create vectors
        # Persist the vectors locally on disk
        # Load from local storage
        # Load your persisted vector here, it could be from a file, database, etc.
        # Define dimensions for GRU input and output
        # gru_input_size = embeddings.size(1)  # Input size is the size of the persisted vector
        gru_input_size = len(indexed_tokens)
        print(gru_input_size)
        gru_hidden_size = 256  # Adjust as needed
        gru_output_size = 1  # Adjust as needed
        # Instantiate GRU model
        gru_model = GRUModel(gru_input_size, gru_hidden_size, output_size=gru_output_size)
        # Initialize the GRU model weights
        # (You can skip this step if you're loading pre-trained weights)
        torch.nn.init.xavier_uniform_(gru_model.gru.weight_ih_l0)
        torch.nn.init.xavier_uniform_(gru_model.gru.weight_hh_l0)
        # Assuming 'embeddings' contains the BERT embeddings obtained from the PDF text
        # Assuming input_tensor is your input tensor of type torch.int64
        input_tensor = input_tensor.to(torch.float32)
        gru_output = gru_model(input_tensor)
        print("GRU output:", gru_output)
        return gru_output

