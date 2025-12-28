import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

__all__ = [
    "read_data",
    "preprocessing",
    "cleaning_text",
    "tokenizing",
    "stemming",
    "remove_stopward",
    "normalize_text",
    "normalize_slang",
]

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

nltk.download('stopwords')
nltk.download('punkt_tab')
_stop_words = set(stopwords.words('indonesian'))

def read_data(file):
    data = pd.read_csv(file)
    return data

def preprocessing(text):
    text = cleaning_text(text)
    text = normalize_text(text)
    text = tokenizing(text)
    text = stemming(text)
    text = remove_stopward(text)
    return text

def cleaning_text(teks:str)->str:
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+|https\S+", '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks) #hapus tanda baca (seperti !)
    teks = re.sub(r'http\S+|@\w+|#\w+', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks)
    teks = re.sub(r'(.)\1{2,}', r'\1', teks) # Repeated Character Normalization
    return teks

def tokenizing(teks ):
    """ Membagi sebuah kalimat menjadi kata satu per satu """
    text = word_tokenize(teks)
    return text

def stemming(text):
    """ Menghilangkan imbuhan akhir dari tiap kata """
    text = _stemmer.stem(text)
    return text

def remove_stopward(text):
    """ Menghilangkan kata yang kurang memiliki makna seperti kata sambung """

    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in _stop_words]
    return  ' '.join(filtered_text)

def normalize_text(text:str)->str:
    text = normalize_slang(text)
    return text

def load_slang_dict(path: str):
    # dropna = remove missing value from dataframe
    # dict = key value pairs object
    df = pd.read_csv(path)
    df = df.dropna()
    return {
        str(slang).strip(): str(formal).strip()
        for slang, formal in zip(df["slang"], df["formal"])
    }

_load_data = load_slang_dict("data/slang.csv")

def normalize_slang(text: str) -> str:
    """ Mengubah kata slang menjadi kata baku berdasarkan kamus """
    return " ".join(
        _load_data.get(word, word)
        for word in text.split()
    )
