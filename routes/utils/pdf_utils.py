import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
from nltk.corpus import stopwords
from langdetect import detect, DetectorFactory
import spacy

def process_text (text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stopwords(tokens, language):
    stop_words = set(stopwords.words(language)) 
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return filtered_tokens

def perfom_lemmatization(tokens):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return lemmatized_tokens

def detect_language(text): 
     DetectorFactory.seed = 0
     detected_language =  detect(text)
     if(detected_language == 'es'): 
            return 'spanish'
     elif(detected_language == 'en'):
         return 'english'
     else:
         return 'english'
     
def test(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    print(len(text))
    print(len(doc))
    
    
    
     
    