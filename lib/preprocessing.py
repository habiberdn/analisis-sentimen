import re
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from lib.lexicon import score_to_label,lexicon_score

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

_stop_words = set(stopwords.words('indonesian')) - {
    "tidak", "bukan", "belum", "kurang", "tanpa", "jangan"
}

__all__ = [
    "read_data",
    "preprocessing",
    "cleaning_text",
    "tokenizing",
    "stemming",
    "remove_stopword",
    "normalize_slang",
    "preprocessing_batch",
]

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Preserve negation words for sentiment analysis
_stop_words = set(stopwords.words('indonesian')) - {
    "tidak", "bukan", "belum", "kurang", "tanpa", "jangan"
}

stem_cache = {}

def load_weighted_lexicon(path: str) -> dict:
    df = pd.read_csv(path)
    lexicon = {}

    for _, row in df.iterrows():
        term = str(row["word"]).lower().strip()
        term = re.sub(r"[^\w\s]", "", term)
        weight = int(row["weight"])
        lexicon[term] = weight

    return lexicon

lexicon = {}
lexicon.update(load_weighted_lexicon("data/lexicon_positive_ver1.csv"))
lexicon.update(load_weighted_lexicon("data/lexicon_negative_ver1.csv"))

def load_slang_dict(path: str) -> dict:
    """Load slang dictionary from CSV"""
    df = pd.read_csv(path)
    df = df.dropna()
    return {
        str(slang).strip().lower(): str(formal).strip().lower()
        for slang, formal in zip(df["slang"], df["formal"])
    }

_slang_dict = load_slang_dict("data/slang(1).csv")

def read_data(file):
    df = pd.read_csv(file)
    df = (
        df.drop_duplicates(subset="full_text")
          .dropna(subset=["full_text"])
          .reset_index(drop=True)
    )
    return df

@st.cache_data
def preprocessing(text):
    """
    Single text preprocessing pipeline

    Sequence:
        1. Cleaning (lowercase, remove URLs, special chars)
        2. Normalization (slang → formal)
        3. Tokenization
        4. Stemming
        5. Stopword Removal
    """
    cleaned = cleaning_text(text)
    normalized = normalize_slang(cleaned)
    tokenized = tokenizing(normalized)
    stemmed = stemming(tokenized)
    stopword_removed = remove_stopword(stemmed)

    return {
        "cleaned": cleaned,
        "normalized": normalized,
        "tokenized": tokenized,
        "stemmed": stemmed,
        "stopword": stopword_removed,
    }

@st.cache_data
def preprocessing_batch(df: pd.DataFrame,text_column: str = 'full_text') -> pd.DataFrame:
    """
       Optimized batch preprocessing for entire DataFrame.
       Much faster than row-by-row processing.
    """
    df['cleaned'] = df[text_column].astype(str).str.lower()
    df['cleaned'] = df['cleaned'].str.replace(r"http\S+|www\S+|https\S+", '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'@\w+|#\w+', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'[^a-z\s]', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'(.)\1{2,}', r'\1', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'\s+', ' ', regex=True).str.strip()

    df = df.drop_duplicates(subset="cleaned").dropna(subset=["cleaned"]).reset_index(drop=True)

    print("Step 2/6: Normalizing slang...")

    df['normalized'] = df['cleaned'].apply(
        lambda text: ' '.join(_slang_dict.get(w, w) for w in text.split())
    )
    
    df['lexicon_score'] = df['normalized'].apply(
            lambda text: lexicon_score(text, lexicon) 
        )
    df['target'] = df['lexicon_score'].apply(score_to_label)

    print("Step 3/6: Tokenizing...")
    df['tokenized'] = df['normalized'].str.split()

    print("Step 4/6: Stopword...")
    df['stopword'] = df['tokenized'].apply(remove_stopword)

    print("Step 5/6: Stemming...")
    def batch_stem(words: list) -> str:
        if words not in stem_cache:
               stem_cache[words] = _stemmer.stem(words)
        return stem_cache[words]

    def stemming_cached(text):
        return ' '.join([batch_stem(w) for w in text])

    df['stemmed'] = df['stopword'].apply(stemming_cached)

    return df

def cleaning_text(teks: str) -> str:
    """Remove URLs, mentions, hashtags, special characters, and normalize repeated chars"""
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+|https\S+|@\w+|#\w+", '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'(.)\1{2,}', r'\1', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def tokenizing(teks):
    """Split text into individual words"""
    text = word_tokenize(teks)
    return  " ".join(text)

def stemming(text):
    """Remove affixes from words using Sastrawi stemmer"""
    stemmed = [_stemmer.stem(word) for word in text]
    return " ".join(stemmed)

def remove_stopword(text):
    """Remove common words that don't carry sentiment (except negations)"""
    filtered_text = [word for word in text if word not in _stop_words]
    return ' '.join(filtered_text)

def normalize_slang(text: str) -> str:
    """Convert slang words to formal Indonesian based on dictionary"""
    return " ".join(
        _slang_dict.get(word, word)
        for word in text.split()
    )
