from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from utils.pdf_utils import (test, extract_text_from_pdf, get_text_chunks, get_embeddings, get_conversation_chain, 
handle_user_input, test2, save_conversation_chain, generate_embedding)
from dotenv import load_dotenv
import os
from langchain.llms import openai, llamacpp
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.db import conversations_collection
from typing import Annotated
from fastapi.security import OAuth2PasswordBearer
from routes.users import get_current_user
from utils.token import decode_access_token
import jwt
from bson import ObjectId
from config.db import users_collection
import chromadb

load_dotenv()
pdfRouter = APIRouter()
API_KEY = os.getenv( "OPENAI_API_KEY")
llm_openai = openai.OpenAI(model="text-davinci-003", api_key=API_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
client = chromadb.PersistentClient()
collection = client.get_or_create_collection(name="pdf",  metadata={"hnsw:space": "cosine"})

class QuestionData(BaseModel):
    question: str
    
class ChatRequest(BaseModel):
    conversation_id: str
    message: str

@pdfRouter.get("/")
def read_root():
    return {"Hello": "World"}

@pdfRouter.post("/upload")
async def upload_pdf(token: Annotated[str, Depends(oauth2_scheme)], file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf(file.file)
        chunks = get_text_chunks(text)
        embeddings = generate_embedding(chunks)
        embeddingList = []
        for embedding in embeddings:
            embeddingList.append(embedding.embedding)
        new_conversation = {
            "filename": file.filename,
            "text": text,
            "embedding": embeddingList
        }
        conversationId = conversations_collection.insert_one(new_conversation).inserted_id
        ids = []
        for id in range(len(embeddingList)):
            ids.append(str(conversationId) + "-" + str(id))            
        collection.add(
            documents=chunks,
            embeddings=embeddingList,
            ids=ids
        )
        conversations_collection.find_one_and_update({"_id": conversationId}, {"$set": {"ids": ids}})
        user = get_current_user(token)
        users_collection.update_one({"_id": ObjectId(user["id"])}, {"$push": {"chats": str(conversationId)}})
        conversationInserted = conversations_collection.find_one({"_id": conversationId})
        
        return str(conversationInserted)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.get("/conversation/{id}")
async def get_conversation(id: str):
    try:
        object_id = ObjectId(id)
        conversation = conversations_collection.find_one({"_id": object_id})
        
        return str(conversation)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.post("/makeQuestion")
async def make_question(data: QuestionData):
    try:
        respuesta = handle_user_input(data.question)
        print(respuesta)
        return {"respuesta": respuesta}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.post("/chat",)
async def chat():
    try:
        test()
        return {"test": "ok"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    
@pdfRouter.post("/test")
async def open_ai(token: Annotated[str, Depends(oauth2_scheme)]):
    try:
        data = get_current_user(token)
        
        
        return {"test": token}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    

