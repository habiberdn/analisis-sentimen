def lexicon_score(text: str, lexicon: dict) -> int:
    score = 0
    text = text.lower()

    for term, weight in lexicon.items():
        if " " in term:  # frasa
            if term in text:
                score += weight
        else:  # kata tunggal
            score += text.split().count(term) * weight

    return score


def score_to_label(score: int) -> str:
    if score > 2:
        return "positif"
    elif score < -2:
        return "negatif"
    else:
        return "netral"
