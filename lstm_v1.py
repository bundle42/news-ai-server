import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def predict(train_df: pd.DataFrame, inference_df: pd.DataFrame):

    # ----------------------------------
    # 0) return feature 추가 (🔥 핵심)
    # ----------------------------------
    train_df["return"] = train_df["close"].pct_change()
    inference_df["return"] = inference_df["close"].pct_change()

    train_df["return"] = train_df["return"].fillna(0)
    inference_df["return"] = inference_df["return"].fillna(0)

    # ----------------------------------
    # 1) feature 컬럼 정의 (🔥 close 제거)
    # ----------------------------------
    feature_cols = [
        "open", "high", "low", "volume",
        "return",

        # 뉴스
        "news_count",
        "sentiment_sum",
        "sentiment_mean",
        "sentiment_std",
        "positive_count",
        "negative_count",
        "neutral_count",
        "positive_ratio",
        "negative_ratio",
        "neutral_ratio",
        "confidence_mean"
    ]

    target_col = "target_close"

    # 실제 존재하는 컬럼만 사용
    feature_cols = [col for col in feature_cols if col in train_df.columns]

    if len(feature_cols) == 0:
        return {"error": "사용 가능한 feature 컬럼이 없습니다."}

    # ----------------------------------
    # 2) 날짜 정렬
    # ----------------------------------
    train_df = train_df.sort_values("date").reset_index(drop=True)

    # ----------------------------------
    # 3) 결측 처리
    # ----------------------------------
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    train_df[target_col] = train_df[target_col].ffill()

    # ----------------------------------
    # 4) 시퀀스 생성 함수
    # ----------------------------------
    def create_sequences(X, y=None, seq_len=5):
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

    SEQ_LEN = 5

    if len(train_df) <= SEQ_LEN:
        return {"error": f"학습 데이터가 너무 적습니다. 최소 {SEQ_LEN + 1}개 이상 필요합니다."}

    # ----------------------------------
    # 5) 학습/검증 분리
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

    if len(X_train) == 0:
        return {"error": "시퀀스 생성 후 학습 데이터가 없습니다."}

    if len(X_val) == 0:
        return {"error": "시퀀스 생성 후 검증 데이터가 없습니다."}

    # ----------------------------------
    # 8) LSTM 모델
    # ----------------------------------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(feature_cols))),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # ----------------------------------
    # 9) 학습
    # ----------------------------------
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
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
    # 11) 전체 예측 (historical)
    # ----------------------------------
    X_all_scaled = feature_scaler.transform(train_df[feature_cols])
    y_all_scaled = target_scaler.transform(train_df[[target_col]])

    X_all_seq, _ = create_sequences(X_all_scaled, y_all_scaled, SEQ_LEN)

    train_pred_scaled = model.predict(X_all_seq, verbose=0)
    train_pred = target_scaler.inverse_transform(train_pred_scaled).flatten()

    train_dates = train_df["date"].reset_index(drop=True)
    infer_dates = inference_df["date"].sort_values().reset_index(drop=True)

    historical_predictions = []
    for i in range(len(train_pred)):
        current_date = train_dates.iloc[SEQ_LEN + i]

        future_dates = [d for d in train_dates if d > current_date]
        if future_dates:
            pred_date = future_dates[0]
        else:
            future_infer = [d for d in infer_dates if d > current_date]
            pred_date = future_infer[0] if future_infer else current_date

        actual_idx = SEQ_LEN + i
        if actual_idx >= len(train_df):
            break

        historical_predictions.append({
            "date": str(current_date.date()),
            "actual_target_close": float(train_df["target_close"].iloc[actual_idx]),
            "predicted_target_close": float(train_pred[i])
        })

    # ----------------------------------
    # 12) 다음 날 예측
    # ----------------------------------
    inference_df = inference_df.sort_values("date").reset_index(drop=True)
    inference_df[feature_cols] = inference_df[feature_cols].fillna(0)

    if len(inference_df) < SEQ_LEN:
        return {"error": "추론 데이터 부족"}

    X_infer_scaled = feature_scaler.transform(inference_df[feature_cols])
    latest_seq = np.expand_dims(X_infer_scaled[-SEQ_LEN:], axis=0)

    next_close_scaled = model.predict(latest_seq, verbose=0)
    next_close_pred = float(target_scaler.inverse_transform(next_close_scaled)[0][0])

    latest_close = float(inference_df["close"].iloc[-1])

    predicted_change = next_close_pred - latest_close
    predicted_change_pct = (predicted_change / latest_close) * 100 if latest_close != 0 else 0.0
    direction = "UP" if next_close_pred > latest_close else "DOWN"

    # ----------------------------------
    # 13) 결과 반환
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
            "predicted_next_close": next_close_pred,
            "predicted_change": float(predicted_change),
            "predicted_change_pct": float(predicted_change_pct),
            "direction": direction
        },

        "historical_predictions": historical_predictions[-30:]
    }