from transformers import pipeline

# 한국어 모델 추천 (처음 실행 시 다운로드됨) 서울대 NLP
sentiment_pipeline = pipeline(
    "text-classification",
    model="snunlp/KR-FinBert-SC"
)

def load_analyze(text):
    result = sentiment_pipeline(text)

    label = result[0]["label"]
    confidence = result[0]["score"]

    if label == "positive":
        sentiment_score = confidence
    elif label == "negative":
        sentiment_score = -confidence
    else:  # neutral
        sentiment_score = 0.0

    return {
        "label": label,
        "confidence": confidence,
        "sentiment_score": sentiment_score
    }