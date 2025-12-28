import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

def preprocessing(text):
    text = cleaning_text(text)
    text = normalize_text(text)
    text = tokenizing(text)
    text = filtering(text)
    text = stemming(text)
    text = remove_stopward(text)
    return text

def cleaning_text(teks:str):
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+|https\S+", '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks) #hapus tanda baca (seperti !)
    teks = re.sub(r'http\S+|@\w+|#\w+', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks)
    teks = re.sub(r'(.)\1{2,}', r'\1', teks) # Repeated Character Normalization
    teks = teks.translate(str.maketrans('', '', string.punctuation))
    teks = ' '.join([kata for kata in teks.split() if kata not in stop_words])
    return teks

def tokenizing(text):
    text = word_tokenize(text)
    return text

def filtering(text):
    clean_words = []
    for word in text:
        if word not in stop_words:
            clean_words.append(word)
    return " ".join(clean_words)

def stemming(text):
    text = stemmer.stem(text)
    return text

def remove_stopward(text):
    stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return  ''.join(filtered_text)

def normalize_text(text):
    text = normalize_slang(text)
    return text

def load_slang_dict(path: str) -> dict[str, str]:
    # dropna = remove missing value from dataframe
    df = pd.read_csv(path)
    df = df.dropna()
    return {
        str(slang).strip(): str(formal).strip()
        for slang, formal in zip(df["slang"], df["formal"])
    }

load_data = load_slang_dict("data/slang.csv")

def normalize_slang(text: str) -> str:
    return " ".join(
        load_data.get(word, word)
        for word in text.split()
    )
