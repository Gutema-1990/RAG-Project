from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pytesseract
from pdf2image import convert_from_path
import os
from langchain.schema import Document

DATA_PATH = "E:/RAG-last/Data"
DB_FAISS_PATH = "E:/RAG-last/vector_xx"

# Function to extract text from images using pytesseract
def extract_text_from_images(images):
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Function to process scanned PDFs and extract text using OCR
def process_scanned_pdfs(file_path):
    images = convert_from_path(file_path)
    text = extract_text_from_images(images)
    return Document(page_content=text, metadata={"source": file_path})

# create a vector db
def create_vector_database():
    # Collecting all PDF files from the directory and its subdirectories
    pdf_files = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    documents = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
        
        # Process scanned PDFs with OCR
        doc = process_scanned_pdfs(file_path)
        documents.append(doc)
    
   
    text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitters.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2', model_kwargs={'device': 'cpu'})
 
    db = FAISS.from_documents(texts, embeddings)

    db.save_local(DB_FAISS_PATH)
  
if __name__ == '__main__':
    create_vector_database()
