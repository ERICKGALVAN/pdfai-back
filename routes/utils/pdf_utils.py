import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
import spacy
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""    
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0, 
        separators=[" ", ",", "\n"]
    )
    chunks = splitter.split_text(text)
    return chunks

     
def test(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    print(len(text))
    print(len(doc))
    
    
    
     
    