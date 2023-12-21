from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from routes.utils.pdf_utils import test, extract_text_from_pdf, get_text_chunks, get_embeddings, get_conversation_chain, handle_user_input
from typing import Annotated
from functools import lru_cache
from dotenv import load_dotenv
import os
from langchain.llms import openai, llamacpp
from pydantic import BaseModel

load_dotenv()
pdfRouter = APIRouter()
API_KEY = os.getenv( "OPENAI_API_KEY")
llm_openai = openai.OpenAI(model="text-davinci-003", api_key=API_KEY)

class QuestionData(BaseModel):
    question: str

@pdfRouter.get("/")
def read_root():
    return {"Hello": "World"}

@pdfRouter.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf(file.file)
        chunks = get_text_chunks(text)
        vector_store = get_embeddings(chunks)
        conversation = get_conversation_chain(vector_store)
        respuesta = conversation({'question': 'de que trata el texto?', 'chat_history': []})
        print(respuesta)
        
        return {"filename": file.filename, "text": chunks }
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
    
@pdfRouter.post("/test")
async def open_ai():
    try:
        respuesta = llm_openai("hola como estas ")
        print(respuesta)
        return {"respuesta": respuesta}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    

