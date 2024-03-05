from pymongo import MongoClient
import pymongo
import os
from dotenv import load_dotenv
import gridfs

load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")
conn  = MongoClient(MONGO_URL)
db = conn["pdfai"]
users_collection = db["users"]
conversations_collection = db["conversations"]
embeddings_collection = db["embeddings"]
documents_collection = db["documents"]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "pdf_index"
fs = gridfs.GridFS(db, collection="fs")
try: 
    connection = conn.server_info()
    print("Connected to MongoDB " + str(connection.get("version")))
except pymongo.errors.ServerSelectionTimeoutError as err:
    print(err)
    print("Could not connect to MongoDB")

