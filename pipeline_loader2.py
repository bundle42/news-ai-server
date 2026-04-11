def load_analyze(text):
    positive_words = ["상승", "호재", "성장", "이익", "급등", "개선", "확대"]
    negative_words = ["하락", "악재", "손실", "급락", "위기", "감소", "축소"]

    score = 0

    for word in positive_words:
        if word in text:
            score += 1

    for word in negative_words:
        if word in text:
            score -= 1

    # label 결정
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    # confidence (간단 정규화)
    confidence = min(abs(score) / 5, 1.0)

    # sentiment_score (기존 구조 유지)
    sentiment_score = confidence if label == "positive" else -confidence if label == "negative" else 0.0

    return {
        "label": label,
        "confidence": confidence,
        "sentiment_score": sentiment_score
    }