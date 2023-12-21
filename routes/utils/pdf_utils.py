import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
import spacy
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

API_KEY = os.getenv( "OPENAI_API_KEY")

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

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="conversation_chain",
        vector_store=vector_store,
        similarity_threshold=0.8,
        max_memory_size=100,
        return_messages=True,
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        
    )
    return conversation_chain

def handle_user_input(user_input):
    print(user_input)
    
    return user_input
    
     
def test(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    print(len(text))
    print(len(doc))
    
    
    
     
    