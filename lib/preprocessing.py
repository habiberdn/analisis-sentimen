import re
import nltk
import pandas as pd
import streamlit as st
from functools import lru_cache
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
    "cleaning_text",
    "tokenizing",
    "remove_stopword",
    "normalize_word",
    "preprocessing_batch",
]

stem_cache = {}

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Preserve negation words for sentiment analysis
_stop_words = set(stopwords.words('indonesian')) - {
    "tidak", "bukan", "belum", "kurang", "tanpa", "jangan"
}

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

_slang_dict = load_slang_dict("data/kamus_alay_sorted.csv")

def read_data(file):
    df = pd.read_csv(file)
    df = (
        df.drop_duplicates(subset="full_text")
          .dropna(subset=["full_text"])
          .reset_index(drop=True)
    )
    return df

def normalize_text(text, slang_dict, lexicon):
    normalized_words = []
    score = 0

    slang_get = slang_dict.get
    lexicon_get = lexicon.get

    for w in text:
        norm = slang_get(w, w)
        normalized_words.append(norm)
        score += lexicon_get(norm, 0)

    normalized_text = ' '.join(normalized_words)
    label = score_to_label(score)

    return normalized_text, score, label

@st.cache_data
def preprocessing_batch(df: pd.DataFrame,text_column: str = 'full_text') -> pd.DataFrame:
    """
       Optimized batch preprocessing for entire DataFrame.
       Much faster than row-by-row processing.
    """

    df['case_folding'] = df[text_column].astype(str).str.lower()
    df['cleaned'] = df['case_folding'].str.replace(r"http\S+|www\S+|https\S+", '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'@\w+|#\w+', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'[^a-z\s]', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'(.)\1{2,}', r'\1', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'\s+', ' ', regex=True).str.strip()

    df = df.drop_duplicates(subset="cleaned").dropna(subset=["cleaned"]).reset_index(drop=True)

    print("Normalizing slang...")

    df['normalized'] = df['cleaned'].apply(
        lambda text: ' '.join(normalize_word(w) for w in text.split())
    )
    print("Finish Normalizing slang...")

    df['normalized'].apply(
        lambda text: ' '.join(_slang_dict.get(w, w) for w in text.split())
    )

    df['lexicon_score'] = df['normalized'].apply(
            lambda text: lexicon_score(text, lexicon)
        )
    df['target'] = df['lexicon_score'].apply(score_to_label)

    df['tokenized'] = df['normalized'].apply(tokenizing)

    df['stopword'] = df['tokenized'].apply(remove_stopword)

    print("Stemming...")
    def stem_word(word: str) -> str:
        if word not in stem_cache:
            stem_cache[word] = _stemmer.stem(word)
        return stem_cache[word]

    def stemming_cached(words: list) -> str:
        return [stem_word(word) for word in words]

    df['stemmed'] = df['stopword'].apply(stemming_cached)
    df = df[df['stemmed'] != ""].reset_index(drop=True)
    print("Finish Stemming...")
    return df

def cleaning_text(teks: str) -> str:
    """Remove URLs, mentions, hashtags, remove special characters (non alphabets e.g angka, tanda baca, emoji, simbol)  , and normalize repeated chars"""
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+|https\S+|@\w+|#\w+", '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'(.)\1{2,}', r'\1', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def tokenizing(teks):
    """Split text into individual words"""
    text = word_tokenize(teks)
    return text

def remove_stopword(text):
    return  [word for word in text if word not in _stop_words]

@lru_cache(maxsize=100_000)
def normalize_word(text: str) -> str:
    """Convert slang words to formal Indonesian based on dictionary"""
    return ' '.join(_slang_dict.get(w, w) for w in text.split()
    )


