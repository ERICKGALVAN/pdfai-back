
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationChain
from config.db import conversations_collection
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from config.db import embeddings_collection, ATLAS_VECTOR_SEARCH_INDEX_NAME
from openai import OpenAI
from langchain.vectorstores import VectorStore
from bson import ObjectId

from langchain.vectorstores.faiss import FAISS

API_KEY = os.getenv( "OPENAI_API_KEY")
llm = ChatOpenAI()
client = OpenAI()


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = "" 
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks: list[str]):
    embeddings = OpenAIEmbeddings()
    vector_search = FAISS.from_texts(
        texts=chunks,
        embeddings=embeddings,
    )
    return vector_search

def generate_embedding(chunks: list[str]):
    embedding = client.embeddings.create(input=chunks, model="text-embedding-ada-002")

    return embedding

def get_vector_store(chunks: list[str]):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        embedding=embeddings,
        texts=chunks,
    )
    return vector_store

def get_conversation_chain(vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        vector_store=vector_store,
        similarity_threshold=0.8,
        max_memory_size=100,
        return_messages=True,
        input_key="question",
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,  
        memory=memory 
    )
    
    return conversation_chain

def get_conversation_chain_with_history(vector_store, chat_history: list):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        vector_store=vector_store,
        similarity_threshold=0.8,
        max_memory_size=10000,
        return_messages=True,
        input_key="question",
    )
    for chat in chat_history:
        if chat["by"] == "user":
            memory.chat_memory.add_user_message(chat["text"])
        else:
            memory.chat_memory.add_ai_message(chat["text"])
    conversation_chain = ConversationalRetrievalChain.from_llm(
        retriever=vector_store.as_retriever(),
        llm=llm,  
        memory=memory 
    )
    
    return conversation_chain

def save_conversation_chain(file_name: str, document: str, question: str):
    inserted_id = conversations_collection.insert_one({"file_name": file_name, "document": ObjectId(document), "chat_history": [
        {
            "by": "user",
            "text": question,
        }
    ]}).inserted_id
    chat_history = conversations_collection.find_one({"_id": inserted_id})
    return chat_history
    

def handle_user_input(user_input):
    print(user_input)
    
    return user_input
    
     
def test():
    response = ""
    print(response)
    return response

def test2():
    mem = 0
    return mem
    
    
    
    
    
     
    