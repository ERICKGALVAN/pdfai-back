from fastapi import FastAPI
from routes.pdf import pdfRouter
from routes.users import userRouter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdfRouter , prefix="/pdf", tags=["pdf"])
app.include_router(userRouter , prefix="/user", tags=["user"])


