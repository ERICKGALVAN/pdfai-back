from pymongo import MongoClient
import pymongo
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URL = os.getenv("MONGO_URL")
conn  = MongoClient(MONGO_URL)
db = conn["pdfai"]
users_collection = db["users"]
conversations_collection = db["conversations"]
embeddings_collection = db["embeddings"]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "pdf_index"
try: 
    connection = conn.server_info()
    print("Connected to MongoDB " + str(connection.get("version")))
except pymongo.errors.ServerSelectionTimeoutError as err:
    print(err)
    print("Could not connect to MongoDB")

