from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from utils.pdf_utils import (test, extract_text_from_pdf, get_text_chunks, get_embeddings, test2, save_conversation_chain, generate_embedding, get_vector_store, get_conversation_chain_with_history,
test_models, get_llms, get_bleu_score, get_bertscore_score, get_rouge_score, llms)
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
    test: bool
    reference: str = None
    
class TestData(BaseModel):
    prediction: str
    reference: str

    
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
async def make_question(data: QuestionData):
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
            conversation_chain = get_conversation_chain_with_history(vector_store, new_chat_history["chat_history"], data.llm, False)
            response = conversation_chain({"question": question})
            if data.llm == 'mistral':
                res = response["answer"].split('Helpful Answer:')[1]
            else:
                res = response["answer"]
            new_chat_history["chat_history"].append({"by": "ai", "text": res})
            conversations_collection.update_one({"file_name": doc["filename"]}, {"$set": {"chat_history": new_chat_history["chat_history"]}})
            test_result = None
            if data.reference != '':
                test_result = await test_question(data.reference, vector_store, new_chat_history, question)
            return {"chat_history": new_chat_history["chat_history"], "test": test_result}
        else:
            chat_history["chat_history"].append({"by": "user", "text": question})
            conversations_collection.update_one({"file_name": doc["filename"]}, {"$set": {"chat_history": chat_history["chat_history"]}})    
            conversation_chain = get_conversation_chain_with_history(vector_store, chat_history["chat_history"], data.llm, False)
            response = conversation_chain({"question": question})
            if data.llm == 'mistral':
                res = response["answer"].split('Helpful Answer:')[1]
            else:
                res = response["answer"]
            chat_history["chat_history"].append({"by": "ai", "text": res})
            
            conversations_collection.update_one({"file_name": doc["filename"]}, {"$set": {"chat_history": chat_history["chat_history"]}})
            test_result = None
            if data.reference != '':
                test_result = await test_question(data.reference, vector_store, chat_history, question)
            return {"chat_history": chat_history["chat_history"], "test": test_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.post("/testQuestion")
async def test_question(reference: str, vector_store, chat_history, question: str):
    results = []
    for llm in llms:
        conv = get_conversation_chain_with_history(vector_store, chat_history["chat_history"], llm, True)
        res = conv({"question": question})
        if llm == 'mistral':
            predition = res["answer"].split('Helpful Answer:')[1]
        else:
            predition = res["answer"]
        predition_list = [predition]
        reference_list = [reference]
        bleu_precision = get_bleu_score(predictions=predition_list, reference=reference_list)
        bert_score = get_bertscore_score(predictions=predition_list, reference=reference_list)
        rouge_score = get_rouge_score(predictions=predition_list, reference=reference_list)
        results.append({"llm": llm, "bleu": bleu_precision, "bert": bert_score, "rouge": rouge_score}) 
    return results
    
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
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@pdfRouter.get("/llms")
async def get_all_llms():
    try:
        llms = get_llms()
        return {"llms": llms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

