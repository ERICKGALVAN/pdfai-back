import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
import spacy
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from typing import AsyncGenerator
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationChain
import asyncio
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

API_KEY = os.getenv( "OPENAI_API_KEY")

class TestClass:
    def __init__(self):
        self.memories = {}
        
    def generate_response(self, conversation_id: str, message: str):
        llm = ChatOpenAI(
            streaming=True,
            temperature=0.0,
            openai_api_key=API_KEY,
        )
        memory = self.memories.get(conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="history",
                return_messages=True,
                )
            self.memories[conversation_id] = memory
            
        conversation_chain = ConversationChain(
            memory=memory,
            prompt=CHAT_PROMPT_TEMPLATE,
            llm=llm,
        )
        result = conversation_chain({"input": message})
        return result        
    
        
testClass = TestClass()

class StreamingConversationChain:
    """
    Class for handling streaming conversation chains.
    It creates and stores memory for each conversation,
    and generates responses using the ChatOpenAI model from LangChain.
    """

    def __init__(self, openai_api_key: str, temperature: float = 0.0):
        self.memories = {}
        self.openai_api_key = openai_api_key
        self.temperature = temperature

    async def generate_response(
        self, conversation_id: str, message: str
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronous function to generate a response for a conversation.
        It creates a new conversation chain for each message and uses a
        callback handler to stream responses as they're generated.
        :param conversation_id: The ID of the conversation.
        :param message: The message from the user.
        """
        callback_handler = AsyncIteratorCallbackHandler()
        llm = ChatOpenAI(
            # callbacks=[callback_handler],
            streaming=True,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
        )

        memory = self.memories.get(conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            self.memories[conversation_id] = memory

        chain = ConversationChain(
            memory=memory,
            prompt=CHAT_PROMPT_TEMPLATE,
            llm=llm,
        )

        chain("question", message)
        
        return chain
        
class ChatRequest(BaseModel):
    """Request model for chat requests.
    Includes the conversation ID and the message from the user.
    """

    conversation_id: str
    message: str
    
CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "You're a AI that knows everything about cats."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

streaming_conversation_chain = StreamingConversationChain(
    openai_api_key=API_KEY,
    temperature=0.0,
)

async def generate_response(data: ChatRequest) -> StreamingResponse:
    """Endpoint for chat requests.
    It uses the StreamingConversationChain instance to generate responses,
    and then sends these responses as a streaming response.
    :param data: The request data.
    """
    response = StreamingResponse(
        streaming_conversation_chain.generate_response(
            data.conversation_id, data.message
        ),
        media_type="text/event-stream",
    )
    print(response)
    return response



def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""    
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        vector_store=vector_store,
        similarity_threshold=0.8,
        max_memory_size=100,
        return_messages=True,
        input_key="question",
    )
   
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    result = conversation_chain({"question": "de que habla el texto?"})
    print(result)
    print("////////////////////")
    print(conversation_chain.memory)
    print("////////////////////")
    print(memory)
    return conversation_chain

def handle_user_input(user_input):
    print(user_input)
    
    return user_input
    
     
def test():
    response = testClass.generate_response("1", "dime como me llamo")
    print(response)
    return response

def test2():
    mem = testClass.memories
    return mem
    
    
    
    
    
     
    