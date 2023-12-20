from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from routes.utils.pdf_utils import test, extract_text_from_pdf, get_text_chunks
from typing import Annotated
from functools import lru_cache
from dotenv import load_dotenv
import os
from langchain.llms import openai, llamacpp

load_dotenv()
pdfRouter = APIRouter()
API_KEY = os.getenv( "OPENAI_API_KEY")
llm_openai = openai.OpenAI(model="text-davinci-003", api_key=API_KEY)

@pdfRouter.get("/")
def read_root():
    return {"Hello": "World"}

@pdfRouter.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf(file.file)
        chunks = get_text_chunks(text)
        print(len(chunks))
        
        # language = detect_language(text)
        # tokens = process_text(text)
        # filtered_tokens = remove_stopwords(tokens, language)
        # lemmatized_tokens = perfom_lemmatization(filtered_tokens)
        # clean_text = " ".join(lemmatized_tokens)
        # test(clean_text)
        return {"filename": file.filename, "text": chunks}
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
    
    

