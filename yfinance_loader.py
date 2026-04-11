import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date

def download_stock_data(stock_name):
    ticker_map = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "현대차": "005380.KS"
    }

    ticker = ticker_map.get(stock_name)

    if ticker is None:
        raise ValueError(f"지원하지 않는 종목입니다: {stock_name}")

    start_date = datetime(2025, 11, 24)
    end_date = date.today() - timedelta(days=1)
    # end_date = datetime(2026, 3, 26)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    # 멀티인덱스 컬럼 처리
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # 컬럼명 정리
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    # 날짜 형식 통일
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    return df