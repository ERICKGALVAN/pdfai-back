from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    _id: str | None = None
    username: str 
    password: str
    chats: list[str] | None = []
    
class UserLogin(BaseModel):
    username: str 
    password: str