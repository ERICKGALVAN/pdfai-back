def conversationEntity(item) -> dict:
    return{
        "id": str(item["_id"]),
        "file_name": item["file_name"],
        "document": str(item["document"]),
        "chat_history": item["chat_history"]
    }
    