from fastapi import FastAPI, APIRouter, HTTPException
from models.user import User, UserLogin
from schemas.userSchema import userEntity, userEntityList
from config.db import users_collection
from bson import ObjectId
from utils.password_utils import verify_password, hash_password
from utils.token import create_access_token, decode_access_token

userRouter = APIRouter()

@userRouter.get("/")
async def get_users():
    users = users_collection.find()
    usersList = userEntityList(users)
    return usersList

@userRouter.post("/")
async def create_user(user: User):
    new_user = dict(user)
    exists = users_collection.find_one({"username": new_user["username"]})
    if exists is not None:
        raise HTTPException(status_code=400, detail="User already exists")
    new_user["password"] = hash_password(new_user["password"])
    id =  users_collection.insert_one(new_user).inserted_id
    user = users_collection.find_one({"_id": ObjectId(id)})
    userInfo = {
        "id": str(user["_id"]),
        "username": user["username"],
        "chats": user["chats"]
    }
    return userInfo

    
@userRouter.get("/{id}")
async def get_user(id: str):
    user = users_collection.find_one({"_id": ObjectId(id)})
    if user is not None:
        return userEntity(user)
    else:
        raise HTTPException(status_code=404, detail="User not found")
    
@userRouter.post("/login")
async def login_user(user: UserLogin):
    foundUser = users_collection.find_one({"username": user.username})
    if foundUser is None: 
        raise HTTPException(status_code=404, detail="User not found")
    match_password = verify_password(user.password, foundUser["password"])
    if not match_password:
        raise HTTPException(status_code=401, detail="Incorrect password")
    token = create_access_token(data={"sub": str(foundUser["_id"])})
    
    return {"token": token}

    


   
    