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
    "remove_stopword",
    "normalize_text",
    "normalize_slang",
    "preprocessing_batch"
]

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)

_stop_words = set(stopwords.words('indonesian'))

# Load slang dictionary once
def load_slang_dict(path: str) -> dict:
    """Load slang dictionary from CSV"""
    df = pd.read_csv(path)
    df = df.dropna()
    return {
        str(slang).strip().lower(): str(formal).strip().lower()
        for slang, formal in zip(df["slang"], df["formal"])
    }

_slang_dict = load_slang_dict("data/slang.csv")

def read_data(file):
    return pd.read_csv(file)

def preprocessing(text):
    # Untuk satu text
    """
    Sequence :
        1. Cleaning
        2. Normalization (slang → formal)
        3. Tokenization
        4. Stemming
        5. Stopword Removal
    """
    cleaned = cleaning_text(text)
    normalized = normalize_text(cleaned)
    tokenized = tokenizing(normalized)
    stemmed = stemming(tokenized)
    stopword = remove_stopword(stemmed)

    return {
            "cleaned": cleaned,
            "normalized": normalized,
            "tokenized": tokenized,
            "stopword": stopword,
            "stemmed": stemmed,
        }

def preprocessing_batch(df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
    """
    Optimized batch preprocessing for entire DataFrame.
    Much faster than applying row by row. (For multiple text)
    """
    # Vectorized operations where possible
    df['cleaned'] = df[text_column].astype(str).str.lower()
    df['cleaned'] = df['cleaned'].str.replace(r"http\S+|www\S+|https\S+", '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'@\w+|#\w+', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'[^a-z\s]', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'(.)\1{2,}', r'\1', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'\s+', ' ', regex=True).str.strip()

    df['normalized'] = df['cleaned'].apply(normalize_text)
    df['tokenized'] = df['normalized'].apply(tokenizing)
    df['stemmed'] = df['tokenized'].apply(stemming)
    df['stopword'] = df['stemmed'].apply(remove_stopword)
    return df

def cleaning_text(teks:str)->str:
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+|https\S+", '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks) #hapus tanda baca (seperti !)
    teks = re.sub(r'http\S+|@\w+|#\w+', '', teks)
    teks = re.sub(r'[^\w\s]', '', teks)
    teks = re.sub(r'(.)\1{2,}', r'\1', teks) # Repeated Character Normalization
    return teks

def tokenizing(teks):
    """ Membagi sebuah kalimat menjadi kata satu per satu """
    text = word_tokenize(teks)
    return text

def stemming(text):
    """ Menghilangkan imbuhan akhir dari tiap kata """
    words = text.split()
    stemmed_words = [_stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def remove_stopword(text):
    """ Menghilangkan kata yang kurang memiliki makna seperti kata sambung """

    filtered_text = [word for word in text if word not in _stop_words]
    return  ' '.join(filtered_text)

def normalize_text(text:str)->str:
    return normalize_slang(text)


def normalize_slang(text: str) -> str:
    """ Mengubah kata slang menjadi kata baku berdasarkan kamus """
    return " ".join(
        _slang_dict.get(word, word)
        for word in text.split()
    )
