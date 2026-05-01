import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


def predict(train_df: pd.DataFrame, inference_df: pd.DataFrame):

    # ----------------------------------
    # 0) return feature 추가
    # ----------------------------------
    train_df["return"] = train_df["close"].pct_change().fillna(0)
    inference_df["return"] = inference_df["close"].pct_change().fillna(0)

    # ----------------------------------
    # 1) feature 컬럼 정의
    # ----------------------------------
    feature_cols = [
        "open", "high", "low", "volume",
        "return",
        "news_count", "sentiment_sum", "sentiment_mean", "sentiment_std",
        "positive_count", "negative_count", "neutral_count",
        "positive_ratio", "negative_ratio", "neutral_ratio",
        "confidence_mean"
    ]

    target_col = "target_close"

    feature_cols = [col for col in feature_cols if col in train_df.columns]

    if len(feature_cols) == 0:
        return {"error": "사용 가능한 feature 컬럼이 없습니다."}

    # ----------------------------------
    # 2) 날짜 정렬
    # ----------------------------------
    train_df = train_df.sort_values("date").reset_index(drop=True)
    inference_df = inference_df.sort_values("date").reset_index(drop=True)

    # ----------------------------------
    # 3) 결측 처리
    # ----------------------------------
    news_cols = [
        "news_count", "sentiment_sum", "sentiment_mean", "sentiment_std",
        "positive_count", "negative_count", "neutral_count",
        "positive_ratio", "negative_ratio", "neutral_ratio",
        "confidence_mean"
    ]

    # 뉴스는 흐름 유지
    train_df[news_cols] = train_df[news_cols].ffill()
    inference_df[news_cols] = inference_df[news_cols].ffill()

    # 그래도 없는 값은 0
    train_df[news_cols] = train_df[news_cols].fillna(0)
    inference_df[news_cols] = inference_df[news_cols].fillna(0)

    # 전체 feature
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    inference_df[feature_cols] = inference_df[feature_cols].fillna(0)

    train_df[target_col] = train_df[target_col].ffill()

    # ----------------------------------
    # 4) 시퀀스 생성
    # ----------------------------------
    def create_sequences(X, y=None, seq_len=10):
        X_seq, y_seq = [], []

        if y is not None:
            for i in range(seq_len, len(X)):
                X_seq.append(X[i - seq_len:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        else:
            for i in range(seq_len, len(X) + 1):
                X_seq.append(X[i - seq_len:i])
            return np.array(X_seq)

    SEQ_LEN = 10

    if len(train_df) <= SEQ_LEN:
        return {"error": f"학습 데이터 부족 (최소 {SEQ_LEN+1})"}

    # ----------------------------------
    # 5) train/val 분리
    # ----------------------------------
    split_idx = int(len(train_df) * 0.8)

    if len(train_df) - split_idx < 20:
        split_idx = max(SEQ_LEN + 1, len(train_df) - 20)

    train_part = train_df.iloc[:split_idx].reset_index(drop=True)
    val_part = train_df.iloc[split_idx:].reset_index(drop=True)

    # ----------------------------------
    # 6) 스케일링
    # ----------------------------------
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(train_part[feature_cols])
    y_train_scaled = target_scaler.fit_transform(train_part[[target_col]])

    X_val_scaled = feature_scaler.transform(val_part[feature_cols])
    y_val_scaled = target_scaler.transform(val_part[[target_col]])

    # ----------------------------------
    # 7) 시퀀스 생성
    # ----------------------------------
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, SEQ_LEN)

    if len(X_train) == 0 or len(X_val) == 0:
        return {"error": "시퀀스 생성 실패"}

    # ----------------------------------
    # 8) 모델
    # ----------------------------------
    model = Sequential([
        LSTM(16, input_shape=(SEQ_LEN, len(feature_cols))),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # ----------------------------------
    # 9) 학습
    # ----------------------------------
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=8,
        callbacks=[early_stop],
        verbose=0
    )

    # ----------------------------------
    # 10) 검증 평가
    # ----------------------------------
    val_pred_scaled = model.predict(X_val, verbose=0)
    val_pred = target_scaler.inverse_transform(val_pred_scaled)
    y_val_real = target_scaler.inverse_transform(y_val)

    rmse = float(np.sqrt(mean_squared_error(y_val_real, val_pred)))
    mae = float(mean_absolute_error(y_val_real, val_pred))

    # ----------------------------------
    # 11) historical 예측
    # ----------------------------------
    X_all_scaled = feature_scaler.transform(train_df[feature_cols])
    y_all_scaled = target_scaler.transform(train_df[[target_col]])

    X_all_seq, _ = create_sequences(X_all_scaled, y_all_scaled, SEQ_LEN)

    train_pred_scaled = model.predict(X_all_seq, verbose=0)
    train_pred = target_scaler.inverse_transform(train_pred_scaled).flatten()

    train_dates = train_df["date"].reset_index(drop=True)

    historical_predictions = []

    for i in range(len(train_pred)):
        idx = SEQ_LEN + i
        if idx >= len(train_df):
            break

        historical_predictions.append({
            "date": str(train_dates.iloc[idx].date()),
            "actual_target_close": float(train_df["target_close"].iloc[idx]),
            "predicted_target_close": float(train_pred[i])
        })

    # ----------------------------------
    # 12) 다음날 예측
    # ----------------------------------
    X_infer_scaled = feature_scaler.transform(inference_df[feature_cols])

    if len(X_infer_scaled) < SEQ_LEN:
        return {"error": "추론 데이터 부족"}

    latest_seq = np.expand_dims(X_infer_scaled[-SEQ_LEN:], axis=0)

    next_scaled = model.predict(latest_seq, verbose=0)
    next_close = float(target_scaler.inverse_transform(next_scaled)[0][0])

    latest_close = float(inference_df["close"].iloc[-1])

    change = next_close - latest_close
    change_pct = (change / latest_close) * 100 if latest_close != 0 else 0

    # ----------------------------------
    # 13) 반환
    # ----------------------------------
    return {
        "feature_cols": feature_cols,
        "sequence_length": SEQ_LEN,
        "validation_metrics": {
            "rmse": rmse,
            "mae": mae
        },
        "latest_input": {
            "last_date": str(inference_df["date"].iloc[-1].date()),
            "last_close": latest_close
        },
        "next_prediction": {
            "predicted_next_close": next_close,
            "predicted_change": float(change),
            "predicted_change_pct": float(change_pct),
            "direction": "UP" if next_close > latest_close else "DOWN"
        },
        "historical_predictions": historical_predictions[-30:]
    }