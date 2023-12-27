from pydantic import BaseModel

class Conversation(BaseModel):
    _id : str | None = None
    filename: str
    chunks: list[str] 
    vector_store: list[list[float]]
    conversation_chain: list | None = []
    