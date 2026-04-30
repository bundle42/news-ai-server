
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
    contents = request["contents"]  # 리스트 받기

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

    if stock_df is None or stock_df.empty:
        print("❌ 데이터 다운로드 실패")
        return {
            "prediction": None,
            "error": "데이터 없음"
        }
    
    # stock_df 예시 컬럼:
    # ["date", "open", "high", "low", "close", "volume", ...]

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
    # -----------------------------
    stock_df["date"] = pd.to_datetime(stock_df["date"], format="%Y-%m-%d", errors="coerce")
    feature_df["date"] = pd.to_datetime(feature_df["date"], format="%y-%m-%d", errors="coerce")

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
    # 규칙:
    # - 뉴스가 거래일 당일이면 그 날짜 row에 붙임
    # - 뉴스가 휴장일(주말/공휴일)이면 다음 거래일의 직전 거래일 row에 붙임
    #
    # 예:
    # - 금요일 뉴스 -> 금요일 row
    # - 토요일 뉴스 -> 금요일 row
    # - 일요일 뉴스 -> 금요일 row
    # - 수요일 공휴일 뉴스 -> 화요일 row (목요일 예측용)
    # -----------------------------
    def map_news_to_input_date(news_date):
        # news_date 이후 첫 거래일 찾기
        future_trading_days = [d for d in stock_dates if d >= news_date]

        if len(future_trading_days) == 0:
            # 주가 데이터 범위 밖의 미래 뉴스는 처리 불가
            return None

        next_trading_day = future_trading_days[0]

        # 뉴스 날짜가 실제 거래일이면 그 날짜 row에 붙임
        if news_date == next_trading_day:
            return news_date

        # 뉴스 날짜가 휴장일이면 다음 거래일의 직전 거래일 row에 붙임
        return prev_map.get(next_trading_day, None)

    feature_df["input_date"] = feature_df["date"].apply(map_news_to_input_date)
    feature_df = feature_df.dropna(subset=["input_date"]).reset_index(drop=True)

    print("=== feature_df with input_date ===")
    print(feature_df[["date", "input_date", "news_count", "sentiment_mean", "sentiment_sum"]].head(30))

    # -----------------------------
    # 7) 같은 input_date로 몰린 뉴스 집계
    # -----------------------------
    # 주의:
    # 단순 평균이 아니라 "기사 수(news_count)" 가중 평균을 사용
    # -----------------------------
    def aggregate_news(group):
        total_news = group["news_count"].sum()

        if total_news == 0:
            total_news = 1  # 0 division 방지

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
        }

        # 비율은 합산 후 재계산
        result["positive_ratio"] = pos_count / total_news
        result["negative_ratio"] = neg_count / total_news
        result["neutral_ratio"] = neu_count / total_news

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

    merged_df = merged_df.dropna(subset=["close"]).reset_index(drop=True) # yf에서 종가 갱신 늦는 경우 제거

    # 뉴스가 없는 거래일은 0으로 채움
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
    # -----------------------------
    # 오늘 row의 feature로 다음 거래일 종가 예측
    # 즉:
    #   월 row -> 화 close
    #   화 row -> 수 close
    #   금 row -> 월 close
    # -----------------------------
    train_df = merged_df.copy()
    train_df["target_close"] = train_df["close"].shift(-1)

    # 마지막 거래일은 target이 없으므로 제거
    train_df = train_df.dropna(subset=["target_close"]).reset_index(drop=True)

    print("=== train_df ===")
    print(train_df.tail(20))

    # -----------------------------
    # 10) 추론용 데이터 생성
    # -----------------------------
    # 최신 날짜 row까지 유지
    inference_df = merged_df.copy()

    # -----------------------------
    # 11) LSTM 예측
    # -----------------------------
    lstm_result = lstm_v1.predict(
        stock_name=stock_name,
        train_df=train_df,
        inference_df=inference_df,
        model_base_dir=".",
        auto_train_if_missing=True,
    )

    # -----------------------------
    # 11-1) 방향 계산 및 적중 여부 추가
    # -----------------------------
    historical_predictions = lstm_result.get("historical_predictions", [])

    for item in historical_predictions:
        # 날짜 통일
        date = pd.to_datetime(item["date"])
        merged_df["date"] = pd.to_datetime(merged_df["date"])

        # 이전 종가 찾기
        previous_rows = merged_df[merged_df["date"] < date]
        if previous_rows.empty:
            # 이전 거래일이 없으면 merged_df 마지막 종가 사용
            previous_close = merged_df["close"].iloc[-1]
        else:
            previous_close = previous_rows.iloc[-1]["close"]
        item["previous_close"] = previous_close

        # 실제 종가
        actual_close = item["actual_target_close"]

        # 실제 방향
        if actual_close > previous_close:
            item["actual_direction"] = "UP"
        elif actual_close < previous_close:
            item["actual_direction"] = "DOWN"
        else:
            item["actual_direction"] = "SAME"

        # 예측 종가
        predicted_close = item["predicted_target_close"]
        if predicted_close > previous_close:
            item["predicted_direction"] = "UP"
        elif predicted_close < previous_close:
            item["predicted_direction"] = "DOWN"
        else:
            item["predicted_direction"] = "SAME"

        # 방향 적중 여부
        item["direction_match"] = (item["actual_direction"] == item["predicted_direction"])

    lstm_result["historical_predictions"] = historical_predictions

    # -----------------------------
    # 12) 반환용 날짜 문자열 변환
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


@app.post("/train", response_model=None)
async def train(request: Dict[str, Any]):
    """
    저장용 학습 엔드포인트.
    - 프론트/백엔드는 기본적으로 /predict만 호출해도 되지만,
      운영에서는 /train을 배치로 돌리고 /predict는 로드+추론만 하는 형태를 권장.
    """
    stock_name = request.get("stockName")
    features = request.get("features", [])

    if not stock_name:
        return {"error": "stockName이 없습니다."}
    if not features:
        return {"error": "features가 없습니다."}

    stock_df = yfinance_loader.download_stock_data(stock_name).copy()
    if stock_df is None or stock_df.empty:
        return {"error": "데이터 없음"}

    feature_df = pd.DataFrame(features).copy()
    if feature_df.empty:
        return {"error": "뉴스 feature 데이터가 비어 있습니다."}

    stock_df["date"] = pd.to_datetime(stock_df["date"], format="%Y-%m-%d", errors="coerce")
    feature_df["date"] = pd.to_datetime(feature_df["date"], format="%y-%m-%d", errors="coerce")
    stock_df = stock_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    feature_df = feature_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

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

    stock_dates = sorted(stock_df["date"].unique())
    if len(stock_dates) < 2:
        return {"error": "주가 데이터가 너무 적어 학습이 어렵습니다."}

    prev_map = {}
    for i in range(1, len(stock_dates)):
        prev_map[stock_dates[i]] = stock_dates[i - 1]

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
        }

        result["positive_ratio"] = pos_count / total_news
        result["negative_ratio"] = neg_count / total_news
        result["neutral_ratio"] = neu_count / total_news
        return pd.Series(result)

    feature_grouped = (
        feature_df
        .groupby("input_date")
        .apply(aggregate_news)
        .reset_index()
        .rename(columns={"input_date": "date"})
    )

    merged_df = pd.merge(stock_df, feature_grouped, on="date", how="left")
    merged_df = merged_df.dropna(subset=["close"]).reset_index(drop=True)

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

    train_df = merged_df.copy()
    train_df["target_close"] = train_df["close"].shift(-1)
    train_df = train_df.dropna(subset=["target_close"]).reset_index(drop=True)

    return lstm_v1.train_and_save(stock_name=stock_name, train_df=train_df, model_base_dir=".")