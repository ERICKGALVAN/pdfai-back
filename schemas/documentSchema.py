def documentEntity(item)-> dict:
    return{
        "id": str(item["_id"]),
        "filename": item["filename"],
        "chunks": item["chunks"],
        "user": str(item["user"]),
        "fileid": str(item["fileid"])
    }
    