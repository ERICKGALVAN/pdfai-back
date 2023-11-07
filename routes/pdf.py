from fastapi import APIRouter, File, UploadFile
from pypdf import PdfReader

pdfRouter = APIRouter()

@pdfRouter.get("/")
def read_root():
    return {"Hello": "World"}

@pdfRouter.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    print(text)
    return {"filename": file.filename}

