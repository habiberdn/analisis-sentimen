def lexicon_score(text: str, lexicon: dict) -> int:
    score = 0
    for word in text.split():
        score += lexicon.get(word, 0)
    return score


def score_to_label(score: int) -> str:
    if score > 0:
        return "positif"
    elif score < 0:
        return "negatif"
    else:
        return "netral"
