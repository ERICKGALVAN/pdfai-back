def userEntity(item)-> dict:
    return{
        "id": str(item["_id"]),
        "username": item["username"],
        "password": item["password"],
        "chats": item["chats"]
    }
    
def userEntityList(item)-> list:
    return [userEntity(item) for item in item]
