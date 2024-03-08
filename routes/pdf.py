from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from utils.pdf_utils import (test, extract_text_from_pdf, get_text_chunks, get_embeddings, 
handle_user_input, test2, save_conversation_chain, generate_embedding, get_vector_store, get_conversation_chain_with_history,
test_models, get_llms)
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
from bson import ObjectId
from config.db import users_collection, documents_collection, fs
from schemas.documentSchema import documentEntity
from fastapi import Response
from schemas.conversationSchema import conversationEntity

load_dotenv()
pdfRouter = APIRouter()
API_KEY = os.getenv( "OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
llm_openai = openai.OpenAI(model="text-davinci-003", api_key=API_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
class QuestionData(BaseModel):
    question: str
    id: str
    llm: str
    
class ChatRequest(BaseModel):
    conversation_id: str
    message: str

@pdfRouter.get("/")
def read_root():
    return {"Hello": "World"}

@pdfRouter.post("/upload")
async def upload_pdf(token: Annotated[str, Depends(oauth2_scheme)], file: UploadFile = File(...)):
    try:
        #print pdf in bytes
        text = extract_text_from_pdf(file.file)
        chunks = get_text_chunks(text)
        # embeddings = generate_embedding(chunks)
        # vector_store = get_vector_store(chunks)
        # conversation_chain = get_conversation_chain(vector_store)
        # result = conversation_chain({"question": "me llamo erick"})
        # result = conversation_chain({"question": "como me llamo?"})
        # print(result)
        file_id = fs.put(file.file, filename=file.filename,)
        user = get_current_user(token)
        inserted_id = documents_collection.insert_one({"filename": file.filename, "chunks": chunks, "user": ObjectId(user['id']), "fileid": file_id, },  ).inserted_id        
        return {"id": str(inserted_id), "text": text, "chunks": chunks}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.get("/conversation/{id}")
async def get_conversation(id: str ):
    try:
        object_id = ObjectId(id)
        conversation = conversations_collection.find_one({"_id": object_id})
        return str(conversation)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.post("/makeQuestion")
async def make_question(data: QuestionData, token: Annotated[str, Depends(oauth2_scheme)]):
    # test_models()
    print(data)
    try:
        question = data.question
        doc = documentEntity(documents_collection.find_one({"_id": ObjectId(data.id)}))
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = doc["chunks"]
        vector_store = get_vector_store(chunks)
        chat_history = conversations_collection.find_one({"document": ObjectId(doc["id"])})
        if chat_history is None:
            new_chat_history = save_conversation_chain(doc["filename"], doc["id"], question)
            conversation_chain = get_conversation_chain_with_history(vector_store, new_chat_history["chat_history"], data.llm)
            response = conversation_chain({"question": question})
            print(response)
            new_chat_history["chat_history"].append({"by": "ai", "text": response["answer"]})
            conversations_collection.update_one({"file_name": doc["filename"]}, {"$set": {"chat_history": new_chat_history["chat_history"]}})
            return {"chat_history": new_chat_history["chat_history"]}
        else:
            chat_history["chat_history"].append({"by": "user", "text": question})
            conversations_collection.update_one({"file_name": doc["filename"]}, {"$set": {"chat_history": chat_history["chat_history"]}})    
            conversation_chain = get_conversation_chain_with_history(vector_store, chat_history["chat_history"], data.llm)
            response = conversation_chain({"question": question})
            print(response)
            chat_history["chat_history"].append({"by": "ai", "text": response["answer"]})
            conversations_collection.update_one({"file_name": doc["filename"]}, {"$set": {"chat_history": chat_history["chat_history"]}})
            return {"chat_history": chat_history["chat_history"]}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.get("/bytes/{id}")
async def get_bytes(token: Annotated[str, Depends(oauth2_scheme)], id: str):
    user = get_current_user(token)
    document = documentEntity(documents_collection.find_one({"_id": ObjectId(id), "user": ObjectId(user['id'])}))
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    fid = document["fileid"]
    output = fs.get(ObjectId(fid)).read( size=3000000)
    print(output)
    return Response(content="output", media_type="application/pdf")

@pdfRouter.get("chat/{id}")
async def chat(id: str):
    conversation = conversationEntity(conversations_collection.find_one({"document": ObjectId(id)}))
    return {"chat":conversation["chat_history"]}
            

@pdfRouter.post("/chat",)
async def chat():
    try:
        test()
        return {"test": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@pdfRouter.post("/test")
async def open_ai(token: Annotated[str, Depends(oauth2_scheme)]):
    try:
        data = get_current_user(token)
        print(data)
        
        
        return {"test": token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.get("/documents/")
async def get_documents(token: Annotated[str, Depends(oauth2_scheme)]):
    try:
        user = get_current_user(token)
        documents = documents_collection.find({"user": ObjectId(user['id'])})
        list_documents = []
        for document in documents:
            list_documents.append(documentEntity(document))
        return {"documents": list_documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.get("/llms")
async def get_all_llms():
    try:
        llms = get_llms()
        return {"llms": llms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

