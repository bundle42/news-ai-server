from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pipeline_loader2
import yfinance_loader
import lstm_v1
import pandas as pd

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "한국어 뉴스 감성 분석 API입니다. /analyze 엔드포인트로 POST 요청을 보내주세요."}


class NewsRequest(BaseModel):
    content: str


@app.post("/read")
def read_news(request: NewsRequest):
    return {"message": f"뉴스 내용이 수신되었습니다: {request.content[:100]}... (총 {len(request.content)}자)"}


@app.post("/analyze-batch")
def analyze_titles(request: dict):
    contents = request["contents"]

    results = []
    for text in contents:
        text = text[:512]
        result = pipeline_loader2.load_analyze(text)
        results.append({
            "label": result["label"],
            "confidence": result["confidence"],
            "sentiment_score": result["sentiment_score"]
        })

    return results


@app.post("/predict", response_model=None)
async def predict(request: Dict[str, Any]):
    stock_name = request.get("stockName")
    features = request.get("features", [])

    if not stock_name:
        return {"error": "stockName이 없습니다."}

    if not features:
        return {"error": "features가 없습니다."}

    # -----------------------------
    # 1) 주가 데이터 불러오기
    # -----------------------------
    stock_df = yfinance_loader.download_stock_data(stock_name).copy()

    # -----------------------------
    # 2) feature 리스트 -> DataFrame
    # -----------------------------
    feature_df = pd.DataFrame(features).copy()

    if feature_df.empty:
        return {
            "prediction_data": [],
            "lstm_result": None,
            "message": "뉴스 feature 데이터가 비어 있습니다."
        }

    # -----------------------------
    # 3) 날짜 형식 통일
    #    stock_df: "%Y-%m-%d"
    #    feature_df: 실제 포맷에 맞게 수정 필요 (기존 "%y-%m-%d"는 2자리 연도라 주의)
    # -----------------------------
    stock_df["date"] = pd.to_datetime(stock_df["date"], errors="coerce")
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")

    stock_df = stock_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # -----------------------------
    # 4) 숫자형 컬럼 보정
    # -----------------------------
    numeric_cols = [
        "news_count",
        "sentiment_mean",
        "sentiment_std",
        "sentiment_sum",
        "positive_count",
        "negative_count",
        "neutral_count",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
        "confidence_mean"
    ]

    for col in numeric_cols:
        if col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

    feature_df = feature_df.fillna(0)

    # -----------------------------
    # 5) 거래일 리스트 준비
    # -----------------------------
    stock_dates = sorted(stock_df["date"].unique())

    if len(stock_dates) < 2:
        return {
            "prediction_data": [],
            "lstm_result": None,
            "message": "주가 데이터가 너무 적어 학습/예측이 어렵습니다."
        }

    # 거래일 -> 직전 거래일 맵
    prev_map = {}
    for i in range(1, len(stock_dates)):
        prev_map[stock_dates[i]] = stock_dates[i - 1]

    # -----------------------------
    # 6) 뉴스 날짜를 "입력 row 날짜(input_date)"로 매핑
    # -----------------------------
    def map_news_to_input_date(news_date):
        future_trading_days = [d for d in stock_dates if d >= news_date]

        if len(future_trading_days) == 0:
            return None

        next_trading_day = future_trading_days[0]

        if news_date == next_trading_day:
            return news_date

        return prev_map.get(next_trading_day, None)

    feature_df["input_date"] = feature_df["date"].apply(map_news_to_input_date)
    feature_df = feature_df.dropna(subset=["input_date"]).reset_index(drop=True)

    print("=== feature_df with input_date ===")
    print(feature_df[["date", "input_date", "news_count", "sentiment_mean", "sentiment_sum"]].head(30))

    # -----------------------------
    # 7) 같은 input_date로 몰린 뉴스 집계 (기사 수 가중 평균)
    # -----------------------------
    def aggregate_news(group):
        total_news = group["news_count"].sum()
        if total_news == 0:
            total_news = 1

        pos_count = group["positive_count"].sum() if "positive_count" in group else 0
        neg_count = group["negative_count"].sum() if "negative_count" in group else 0
        neu_count = group["neutral_count"].sum() if "neutral_count" in group else 0

        result = {
            "news_count": group["news_count"].sum(),
            "sentiment_sum": group["sentiment_sum"].sum(),
            "sentiment_mean": (group["sentiment_mean"] * group["news_count"]).sum() / total_news,
            "sentiment_std": (group["sentiment_std"] * group["news_count"]).sum() / total_news,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "confidence_mean": (group["confidence_mean"] * group["news_count"]).sum() / total_news,
            "positive_ratio": pos_count / total_news,
            "negative_ratio": neg_count / total_news,
            "neutral_ratio": neu_count / total_news,
        }

        return pd.Series(result)

    feature_grouped = (
        feature_df
        .groupby("input_date")
        .apply(aggregate_news)
        .reset_index()
        .rename(columns={"input_date": "date"})
    )

    print("=== feature_grouped ===")
    print(feature_grouped.head(30))

    # -----------------------------
    # 8) 주가 + 뉴스 병합
    # -----------------------------
    merged_df = pd.merge(
        stock_df,
        feature_grouped,
        on="date",
        how="left"
    )

    news_feature_cols = [
        "news_count", "sentiment_sum", "sentiment_mean", "sentiment_std",
        "positive_count", "negative_count", "neutral_count",
        "positive_ratio", "negative_ratio", "neutral_ratio",
        "confidence_mean"
    ]

    for col in news_feature_cols:
        if col not in merged_df.columns:
            merged_df[col] = 0
        merged_df[col] = merged_df[col].fillna(0)

    print("=== merged_df ===")
    print(merged_df.tail(20))

    # -----------------------------
    # 9) 학습용 데이터 생성
    #    오늘 row의 feature로 다음 거래일 종가 예측
    # -----------------------------
    train_df = merged_df.copy()
    train_df["target_close"] = train_df["close"].shift(-1)
    train_df = train_df.dropna(subset=["target_close"]).reset_index(drop=True)

    print("=== train_df ===")
    print(train_df.tail(20))

    # -----------------------------
    # 10) 추론용 데이터 생성
    # -----------------------------
    inference_df = merged_df.copy()

    # -----------------------------
    # 11) LSTM 예측
    # -----------------------------
    lstm_result = lstm_v1.predict(train_df, inference_df)

    # -----------------------------
    # 12) 방향 계산 및 적중 여부 추가
    # -----------------------------
    historical_predictions = lstm_result.get("historical_predictions", [])

    merged_df["date"] = pd.to_datetime(merged_df["date"])

    for item in historical_predictions:
        date = pd.to_datetime(item["date"])

        previous_rows = merged_df[merged_df["date"] < date]

        # 수정: 이전 거래일이 없으면 자기 자신(첫 행) 종가 사용
        if previous_rows.empty:
            previous_close = float(merged_df["close"].iloc[0])
        else:
            previous_close = float(previous_rows.iloc[-1]["close"])

        item["previous_close"] = previous_close

        actual_close = item["actual_target_close"]
        predicted_close = item["predicted_target_close"]

        if actual_close > previous_close:
            item["actual_direction"] = "UP"
        elif actual_close < previous_close:
            item["actual_direction"] = "DOWN"
        else:
            item["actual_direction"] = "SAME"

        if predicted_close > previous_close:
            item["predicted_direction"] = "UP"
        elif predicted_close < previous_close:
            item["predicted_direction"] = "DOWN"
        else:
            item["predicted_direction"] = "SAME"

        item["direction_match"] = (item["actual_direction"] == item["predicted_direction"])

    lstm_result["historical_predictions"] = historical_predictions

    # -----------------------------
    # 13) 반환용 날짜 문자열 변환
    # -----------------------------
    merged_df_return = merged_df.copy()
    merged_df_return["date"] = merged_df_return["date"].dt.strftime("%Y-%m-%d")

    train_df_return = train_df.copy()
    train_df_return["date"] = train_df_return["date"].dt.strftime("%Y-%m-%d")

    return {
        "stock_name": stock_name,
        "prediction_data": merged_df_return.to_dict(orient="records"),
        "train_data": train_df_return.to_dict(orient="records"),
        "lstm_result": lstm_result
    }