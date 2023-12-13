from fastapi import APIRouter, File, UploadFile, HTTPException
from pypdf import PdfReader
from routes.utils.pdf_utils import process_text, remove_stopwords, perfom_lemmatization, detect_language, test

pdfRouter = APIRouter()

@pdfRouter.get("/")
def read_root():
    return {"Hello": "World"}

@pdfRouter.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        reader = PdfReader(file.file)
        text = ""    
        for page in reader.pages:
            text += page.extract_text()
        language = detect_language(text)
        tokens = process_text(text)
        filtered_tokens = remove_stopwords(tokens, language)
        lemmatized_tokens = perfom_lemmatization(filtered_tokens)
        clean_text = " ".join(lemmatized_tokens)
        test(clean_text)
        return {"filename": file.filename, "text": clean_text}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    

