from pydantic import BaseModel
from typing import Optional

class User(BaseModel):
    _id: Optional[str] = None
    username: str 
    password: str
    chats: Optional[list] = []
    
class UserLogin(BaseModel):
    username: str 
    password: str