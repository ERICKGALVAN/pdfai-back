
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from config.db import conversations_collection
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from config.db import embeddings_collection, ATLAS_VECTOR_SEARCH_INDEX_NAME
from langchain.callbacks import AsyncIteratorCallbackHandler
from openai import OpenAI

from langchain.vectorstores import VectorStore
from bson import ObjectId
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.vectorstores.faiss import FAISS

from langchain_community.llms import llamacpp

load_dotenv()
API_KEY = os.getenv( "OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
llm_openai = ChatOpenAI()
client = OpenAI()

llm_flan = HuggingFaceHub(repo_id="google/flan-t5-base", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
llm_byt5 = HuggingFaceHub(repo_id="google/byt5-small", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
llm_flan_xxl = HuggingFaceHub(repo_id="google/flan-t5-xxl", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
llm_blenderbot = HuggingFaceHub(repo_id="facebook/blenderbot-400M-distill", huggingfacehub_api_token=HUGGINGFACE_API_KEY, model_kwargs={
    "truncation": "only_first",
})
llm_fastchat = HuggingFaceHub(repo_id="lmsys/fastchat-t5-3b-v1.0", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
llm_belle = HuggingFaceHub(repo_id="BelleGroup/BELLE-7B-2M", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
llm_gpt2 = HuggingFaceHub(repo_id="openai-community/gpt2", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
# llm_llama = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", huggingfacehub_api_token=HUGGINGFACE_API_KEY)
llm_mistral = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=HUGGINGFACE_API_KEY)

llms = {
    "gpt": llm_openai,
    "flan": llm_flan,
    # "byt5": llm_byt5,
    # "flan_xxl": llm_flan_xxl,
    # "blenderbot": llm_blenderbot,
    # "fastchat": llm_fastchat,
    # "belle": llm_belle,
    # "gpt2": llm_gpt2,
    "mistral": llm_mistral,
}


from evaluate import load

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#https://huggingface.co/evaluate-metric

# meteor = load('meteor')
# predictions = ["it is about a dog named boby"]
# reference = ["the story of a dog named boby"]
# results = meteor.compute(predictions=predictions, references=reference)
# print("///////HOLA////////")
# print(results)

#https://huggingface.co/spaces/evaluate-metric/bleu
bleu = load('bleu')

# https://huggingface.co/spaces/evaluate-metric/perplexity
perplexity = load('perplexity', module_type="metric")

#https://huggingface.co/spaces/evaluate-metric/bertscore
bertscore = load('bertscore')

# https://huggingface.co/spaces/evaluate-metric/rouge
rouge = load('rouge')

# https://huggingface.co/spaces/evaluate-metric/wiki_split
wiki_split = load('wiki_split')

def get_bleu_score(predictions: list[str], reference:list[str]):
    results = bleu.compute(predictions=predictions, references=reference)
    return results

def get_perplexity_score(predictions: list[str], model_id:str):
    try:
        results = perplexity.compute(predictions=predictions, model_id=model_id)
        return results
    except Exception as e:
        print(e)

def get_bertscore_score(predictions: list[str], reference:list[str]):
    results = bertscore.compute(predictions=predictions, references=reference, lang="en")
    return results

def get_rouge_score(predictions: list[str], reference:list[str]):
    results = rouge.compute(predictions=predictions, references=reference)
    return results

def get_wiki_split_score(predictions: list[str], reference:list[str]):
    try:
        results = wiki_split.compute(predictions=predictions, references=reference, sources=reference)
        return results
    except Exception as e:
        print(e)


def test_models():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", max_new_tokens=100000)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    input_text = "how are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = "" 
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def get_embeddings(chunks: list[str]):
    embeddings = OpenAIEmbeddings()
    vector_search = FAISS.from_texts(
        texts=chunks,
        embeddings=embeddings,
    )
    return vector_search

def generate_embedding(chunks: list[str]):
    embedding = client.embeddings.create(input=chunks, model="text-embedding-ada-002")

    return embedding

def get_vector_store(chunks: list[str]):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        embedding=embeddings,
        texts=chunks,
    )
    return vector_store


def get_conversation_chain_with_history(vector_store, chat_history: list, llm:str, test:bool):
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            vector_store=vector_store,
            similarity_threshold=0.8,
            max_memory_size=1000000,
            return_messages=True,
            input_key="question",
        )
        
        for chat in chat_history:
                if chat["by"] == "user":
                    memory.chat_memory.add_user_message(chat["text"])
                else:
                    memory.chat_memory.add_ai_message(chat["text"])
        conversation_chain = ConversationalRetrievalChain.from_llm(
            retriever=vector_store.as_retriever(),
            llm=llms[llm],
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        raise e
        

def save_conversation_chain(file_name: str, document: str, question: str):
    inserted_id = conversations_collection.insert_one({"file_name": file_name, "document": ObjectId(document), "chat_history": [
        {
            "by": "user",
            "text": question,
        }
    ]}).inserted_id
    chat_history = conversations_collection.find_one({"_id": inserted_id})
    return chat_history
    


     
def test():
    response = ""
    print(response)
    return response

def test2():
    mem = 0
    return mem

def get_llms():
    return list(llms.keys())
    
    
    
    
    
     
    