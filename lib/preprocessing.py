import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from lib.lexicon import lexicon_score, score_to_label

# ============= DATA LOADING =============
positive_df = pd.read_csv("data/positive.csv")
negative_df = pd.read_csv("data/negative.csv")

positive_words = positive_df['word'].str.lower().tolist()
negative_words = negative_df['word'].str.lower().tolist()

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()
# Build lexicon dictionary
lexicon = {}
for word in positive_words:
    stemmed = _stemmer.stem(word)
    lexicon[stemmed] = 1  # Positif

for word in negative_words:
    stemmed = _stemmer.stem(word)
    lexicon[stemmed] = -1

__all__ = [
    "read_data",
    "preprocessing",
    "cleaning_text",
    "tokenizing",
    "stemming",
    "remove_stopword",
    "normalize_slang",
    "preprocessing_batch"
]

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Preserve negation words for sentiment analysis
_stop_words = set(stopwords.words('indonesian')) - {
    "tidak", "bukan", "belum", "kurang", "tanpa", "jangan"
}

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
    df = pd.read_csv(file)
    df = (
        df.drop_duplicates(subset="full_text")
          .dropna(subset=["full_text"])
          .reset_index(drop=True)
    )
    return df

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

def preprocessing_batch(df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
    """
    Optimized batch preprocessing for entire DataFrame.
    Much faster than row-by-row processing.
    """
    # Vectorized cleaning operations
    df['cleaned'] = df[text_column].astype(str).str.lower()
    df['cleaned'] = df['cleaned'].str.replace(r"http\S+|www\S+|https\S+", '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'@\w+|#\w+', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'[^a-z\s]', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'(.)\1{2,}', r'\1', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df = (
        df.drop_duplicates(subset="cleaned")
          .dropna(subset=["cleaned"])
          .reset_index(drop=True)
    )
    df['normalized'] = df['cleaned'].apply(normalize_slang)
    df['tokenized'] = df['normalized'].apply(tokenizing)
    df['stemmed'] = df['tokenized'].apply(stemming)
    df['stopword'] = df['stemmed'].apply(remove_stopword)

    df['lexicon_score'] = df['stopword'].apply(
        lambda x: lexicon_score(x, lexicon)
    )
    df['target'] = df['lexicon_score'].apply(score_to_label)
    df['lexicon_score'].value_counts().sort_index()

    return df

# ============= HELPER FUNCTIONS =============
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
    return word_tokenize(teks)

def stemming(text):
    """Remove affixes from words using Sastrawi stemmer"""
    stemmed = [_stemmer.stem(word) for word in text]
    return " ".join(stemmed)

def remove_stopword(text):
    """Remove common words that don't carry sentiment (except negations)"""
    words = text.split()
    filtered_text = [word for word in words if word not in _stop_words]
    return ' '.join(filtered_text)

def normalize_slang(text: str) -> str:
    """Convert slang words to formal Indonesian based on dictionary"""
    return " ".join(
        _slang_dict.get(word, word)
        for word in text.split()
    )
