import re
import nltk
import pandas as pd
import streamlit as st
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from lib.lexicon import score_to_label, lexicon_score_with_details

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

# Preserve negation words for sentiment analysis
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
    "get_phrase_info",
]

stem_cache = {}


def load_weighted_lexicon(path: str) -> dict:
    """Load lexicon dari CSV dengan akumulasi bobot untuk kata duplikat"""
    df = pd.read_csv(path)
    lexicon = {}

    for _, row in df.iterrows():
        term = str(row["word"]).lower().strip()
        term = re.sub(r"[^\w\s]", "", term)
        weight = int(row["weight"])
        
        # Akumulasi bobot jika kata sudah ada
        if term in lexicon:
            lexicon[term] += weight
        else:
            lexicon[term] = weight

    return lexicon


def load_all_lexicons(positive_path: str, negative_path: str) -> dict:
    """
    Load semua lexicon dengan akumulasi bobot untuk kata duplikat.
    Jika kata 'sudah' ada di kedua file dengan bobot -2 dan 3,
    maka bobot finalnya adalah 1 (-2 + 3).
    """
    lexicon = {}
    
    # Load positive lexicon
    pos_lex = load_weighted_lexicon(positive_path)
    for word, weight in pos_lex.items():
        lexicon[word] = lexicon.get(word, 0) + weight
    
    # Load negative lexicon dengan akumulasi
    neg_lex = load_weighted_lexicon(negative_path)
    for word, weight in neg_lex.items():
        lexicon[word] = lexicon.get(word, 0) + weight
    
    return lexicon


# Load lexicon dengan metode yang diperbaiki
lexicon = load_all_lexicons(
    "data/lexicon_positive_ver1.csv",
    "data/negative_full.csv"
)


def get_phrase_info(lexicon: dict) -> dict:
    """
    Mendapatkan informasi tentang frasa dalam lexicon.
    
    Returns:
        dict: {
            'phrases': list of phrase strings,
            'phrase_count': jumlah frasa,
            'single_word_count': jumlah kata tunggal,
            'phrase_details': list of (phrase, weight) tuples
        }
    """
    phrases = []
    phrase_details = []
    single_words = []
    
    for word, weight in lexicon.items():
        if " " in word:  # Frasa multi-kata
            phrases.append(word)
            phrase_details.append((word, weight))
        else:
            single_words.append(word)
    
    return {
        'phrases': sorted(phrases),
        'phrase_count': len(phrases),
        'single_word_count': len(single_words),
        'phrase_details': sorted(phrase_details, key=lambda x: abs(x[1]), reverse=True)
    }


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
    """Read CSV file and remove duplicates"""
    df = pd.read_csv(file)
    df = (
        df.drop_duplicates(subset="full_text")
          .dropna(subset=["full_text"])
          .reset_index(drop=True)
    )
    return df


def normalize_text(text, slang_dict, lexicon):
    """Normalize text and calculate sentiment score"""
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
def preprocessing_batch(df: pd.DataFrame, text_column: str = 'full_text') -> pd.DataFrame:
    """
    Optimized batch preprocessing for entire DataFrame.
    Much faster than row-by-row processing.
    """
    # Drop unnecessary columns
    columns_to_drop = [
        'conversation_id_str', 'created_at', 'favorite_count', 'id_str',
        'image_url', 'in_reply_to_screen_name', 'lang', 'location',
        'quote_count', 'reply_count', 'retweet_count', 'tweet_url',
        'user_id_str', 'username'
    ]
    df = df.drop(columns=columns_to_drop)

    # Case folding
    df['case_folding'] = df[text_column].astype(str).str.lower()
    
    # Cleaning text
    df['cleaned'] = df['case_folding'].str.replace(r"http\S+|www\S+|https\S+", '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'@\w+|#\w+', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'[^a-z\s]', '', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'(.)\1{2,}', r'\1', regex=True)
    df['cleaned'] = df['cleaned'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Remove duplicates and empty texts
    df = df.drop_duplicates(subset="cleaned").dropna(subset=["cleaned"]).reset_index(drop=True)

    print("Normalizing slang...")
    
    # Normalize slang words
    df['normalized'] = df['cleaned'].apply(
        lambda text: ' '.join(_slang_dict.get(w, w) for w in text.split())
    )
    
    print("Finish Normalizing slang...")

    # Tokenize
    df['tokenized'] = df['normalized'].apply(tokenizing)

    # Remove stopwords
    df['stopword'] = df['tokenized'].apply(remove_stopword)

    # Calculate lexicon score dengan details (termasuk frasa yang ditemukan)
    print("Calculating lexicon scores...")
    score_details = df['normalized'].apply(
        lambda text: lexicon_score_with_details(text, lexicon)
    )
    
    df['lexicon_score'] = score_details.apply(lambda x: x['score'])
    df['found_phrases'] = score_details.apply(lambda x: x['phrases_found'])
    df['found_single_words'] = score_details.apply(lambda x: x['single_words_found'])
    df['target'] = df['lexicon_score'].apply(score_to_label)
    
    print("Stemming...")
    
    # Stemming with cache
    def stem_word(word: str) -> str:
        if word not in stem_cache:
            stem_cache[word] = _stemmer.stem(word)
        return stem_cache[word]

    def stemming_cached(words: list) -> str:
        return ' '.join(stem_word(w) for w in words)

    df['stemmed'] = df['stopword'].apply(stemming_cached)
    
    # Remove empty stemmed texts
    df = df[df['stemmed'] != ""].reset_index(drop=True)
    
    print("Finish Stemming...")
    print("stemmed value:", df['stemmed'].head(10))
    print("stopword value:", df['stopword'].head(10))

    return df


def cleaning_text(teks: str) -> str:
    """Remove URLs, mentions, hashtags, special characters and normalize repeated chars"""
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
    """Remove stopwords from tokenized text"""
    return [word for word in text if word not in _stop_words]


@lru_cache(maxsize=100_000)
def normalize_word(text: str) -> str:
    """Convert slang words to formal Indonesian based on dictionary"""
    return ' '.join(_slang_dict.get(w, w) for w in text.split())