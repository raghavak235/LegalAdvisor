from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# custom_model.py
class CustomLanguageModel:
    def generate_response(self, prompt):
        # Load document using PyPDFLoader document loader
        loader = PyPDFLoader("ipc.pdf")
        documents = loader.load()
        # Split document in chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        docs = text_splitter.split_documents(documents=documents)
        embeddings = OpenAIEmbeddings()
        # Create vectors
        vectorstore = FAISS.from_documents(docs, embeddings)
         # Persist the vectors locally on disk
        vectorstore.save_local("faiss_index_constitution")

         # Load from local storage
        persisted_vectorstore = FAISS.load_local("faiss_index_constitution", embeddings)

        # Use RetrievalQA chain for orchestration
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=persisted_vectorstore.as_retriever())
        result = qa.run(prompt)
        print(result)
        return result

