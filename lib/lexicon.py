"""
Module untuk sentiment analysis menggunakan lexicon-based approach.
Includes phrase detection and detailed scoring information.
"""


def lexicon_score(text: str, lexicon: dict) -> int:
    """
    Hitung skor sentimen berdasarkan lexicon.
    
    Args:
        text: Teks yang akan dianalisis
        lexicon: Dictionary berisi kata/frasa dan bobotnya
    
    Returns:
        int: Total skor sentimen
    """
    score = 0
    words = text.split()
    used_indices = set()
    
    # 1. Proses frasa multi-kata terlebih dahulu
    for phrase, weight in lexicon.items():
        if " " in phrase and phrase in text:
            phrase_words = phrase.split()
            phrase_len = len(phrase_words)
            
            for i in range(len(words) - phrase_len + 1):
                if " ".join(words[i:i+phrase_len]) == phrase:
                    score += weight
                    used_indices.update(range(i, i + phrase_len))
    
    # 2. Proses kata tunggal (skip yang sudah dalam frasa)
    for idx, word in enumerate(words):
        if idx not in used_indices and word in lexicon and " " not in word:
            score += lexicon[word]
    
    return score


def lexicon_score_with_details(text: str, lexicon: dict) -> dict:
    """
    Hitung skor sentimen dengan detail lengkap termasuk frasa yang ditemukan.
    
    Args:
        text: Teks yang akan dianalisis
        lexicon: Dictionary berisi kata/frasa dan bobotnya
    
    Returns:
        dict: {
            'score': int,
            'phrases_found': list of (phrase, weight, position),
            'single_words_found': list of (word, weight, position),
            'total_matches': int
        }
    """
    score = 0
    words = text.split()
    used_indices = set()
    phrases_found = []
    single_words_found = []
    
    # 1. Proses frasa multi-kata terlebih dahulu
    for phrase, weight in lexicon.items():
        if " " in phrase and phrase in text:
            phrase_words = phrase.split()
            phrase_len = len(phrase_words)
            
            for i in range(len(words) - phrase_len + 1):
                if " ".join(words[i:i+phrase_len]) == phrase:
                    score += weight
                    phrases_found.append({
                        'text': phrase,
                        'weight': weight,
                        'position': i,
                        'type': 'positive' if weight > 0 else 'negative' if weight < 0 else 'neutral'
                    })
                    used_indices.update(range(i, i + phrase_len))
    
    # 2. Proses kata tunggal (skip yang sudah dalam frasa)
    for idx, word in enumerate(words):
        if idx not in used_indices and word in lexicon and " " not in word:
            weight = lexicon[word]
            score += weight
            single_words_found.append({
                'text': word,
                'weight': weight,
                'position': idx,
                'type': 'positive' if weight > 0 else 'negative' if weight < 0 else 'neutral'
            })
    
    return {
        'score': score,
        'phrases_found': phrases_found,
        'single_words_found': single_words_found,
        'total_matches': len(phrases_found) + len(single_words_found)
    }


def score_to_label(score: int) -> str:
    """
    Konversi skor menjadi label sentimen.
    
    Args:
        score: Skor sentimen (integer)
    
    Returns:
        str: Label sentimen ('positif', 'negatif', atau 'netral')
    """
    if score > 0:
        return "positif"
    elif score < 0:
        return "negatif"
    else:
        return "netral"


def get_phrase_statistics(lexicon: dict) -> dict:
    """
    Mendapatkan statistik tentang frasa dan kata dalam lexicon.
    
    Args:
        lexicon: Dictionary lexicon
    
    Returns:
        dict: Statistik lengkap tentang lexicon
    """
    phrases = []
    single_words = []
    positive_phrases = []
    negative_phrases = []
    
    for word, weight in lexicon.items():
        if " " in word:  # Frasa multi-kata
            phrases.append((word, weight))
            if weight > 0:
                positive_phrases.append((word, weight))
            elif weight < 0:
                negative_phrases.append((word, weight))
        else:
            single_words.append((word, weight))
    
    return {
        'total_phrases': len(phrases),
        'total_single_words': len(single_words),
        'positive_phrases': sorted(positive_phrases, key=lambda x: x[1], reverse=True),
        'negative_phrases': sorted(negative_phrases, key=lambda x: x[1]),
        'all_phrases': sorted(phrases, key=lambda x: abs(x[1]), reverse=True),
        'phrase_list': sorted([p[0] for p in phrases])
    }


def analyze_text_detailed(text: str, lexicon: dict) -> dict:
    """
    Analisis teks lengkap dengan semua detail.
    
    Args:
        text: Teks yang akan dianalisis
        lexicon: Dictionary lexicon
    
    Returns:
        dict: Hasil analisis lengkap
    """
    details = lexicon_score_with_details(text, lexicon)
    
    return {
        'original_text': text,
        'score': details['score'],
        'label': score_to_label(details['score']),
        'phrases_found': details['phrases_found'],
        'single_words_found': details['single_words_found'],
        'total_matches': details['total_matches'],
        'has_phrases': len(details['phrases_found']) > 0,
        'phrase_count': len(details['phrases_found']),
        'word_count': len(details['single_words_found'])
    }