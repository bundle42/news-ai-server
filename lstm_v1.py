import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from model_store import get_model_dir, load_artifacts, save_artifacts

SEQ_LEN = 5


def _select_feature_cols(train_df: pd.DataFrame) -> List[str]:
    feature_cols = [
        # 주가
        "open", "high", "low", "close", "volume",

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

    return [col for col in feature_cols if col in train_df.columns]


def _create_sequences(X: np.ndarray, y: Optional[np.ndarray] = None, seq_len: int = SEQ_LEN):
    X_seq, y_seq = [], []

    if y is not None:
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    for i in range(seq_len, len(X) + 1):
        X_seq.append(X[i - seq_len:i])
    return np.array(X_seq)


def _build_model(n_features: int) -> Any:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
        Dropout(0.2),

        LSTM(32, return_sequences=False),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )
    return model


def train_and_save(
    *,
    stock_name: str,
    train_df: pd.DataFrame,
    model_base_dir: str = ".",
) -> Dict[str, Any]:
    """
    학습을 1회 수행하고 모델/스케일러를 디스크에 저장.
    - 누수 방지: scaler는 train split에만 fit
    - 반환: 학습 메타 + 검증 지표
    """
    target_col = "target_close"

    feature_cols = _select_feature_cols(train_df)
    if len(feature_cols) == 0:
        return {"error": "사용 가능한 feature 컬럼이 없습니다."}

    train_df = train_df.sort_values("date").reset_index(drop=True)
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    train_df[target_col] = train_df[target_col].ffill()

    if len(train_df) <= SEQ_LEN:
        return {"error": f"학습 데이터가 너무 적습니다. 최소 {SEQ_LEN + 1}개 이상 필요합니다."}

    # 시계열 분리(시간 순서 유지)
    split_row = int(len(train_df) * 0.8)
    if split_row <= SEQ_LEN:
        split_row = SEQ_LEN + 1
    if split_row >= len(train_df) - 1:
        split_row = max(SEQ_LEN + 1, len(train_df) - 2)

    df_train = train_df.iloc[:split_row].reset_index(drop=True)
    df_val = train_df.iloc[split_row:].reset_index(drop=True)

    # scaler는 train 구간으로만 fit (누수 제거)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_scaled_full = feature_scaler.fit_transform(df_train[feature_cols])
    y_train_scaled_full = target_scaler.fit_transform(df_train[[target_col]])

    X_train_seq, y_train_seq = _create_sequences(X_train_scaled_full, y_train_scaled_full, seq_len=SEQ_LEN)

    # val 구간은 train scaler로 transform 후, val에서 만들 수 있는 시퀀스만 평가
    X_val_scaled_full = feature_scaler.transform(df_val[feature_cols])
    y_val_scaled_full = target_scaler.transform(df_val[[target_col]])

    if len(df_val) <= SEQ_LEN:
        # 검증 구간이 너무 짧으면 metrics를 계산하지 않음
        X_val_seq, y_val_seq = None, None
    else:
        X_val_seq, y_val_seq = _create_sequences(X_val_scaled_full, y_val_scaled_full, seq_len=SEQ_LEN)

    model = _build_model(n_features=len(feature_cols))
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    if X_val_seq is None:
        model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=8,
            callbacks=[early_stop],
            verbose=2
        )
        rmse, mae = None, None
    else:
        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=100,
            batch_size=8,
            callbacks=[early_stop],
            verbose=2
        )

        val_pred_scaled = model.predict(X_val_seq, verbose=0)
        val_pred = target_scaler.inverse_transform(val_pred_scaled)
        y_val_real = target_scaler.inverse_transform(y_val_seq)
        rmse = float(np.sqrt(mean_squared_error(y_val_real, val_pred)))
        mae = float(mean_absolute_error(y_val_real, val_pred))

    last_train_date = str(pd.to_datetime(train_df["date"].iloc[-1]).date())

    meta = {
        "stock_name": stock_name,
        "seq_len": SEQ_LEN,
        "feature_cols": feature_cols,
        "trained_until": last_train_date,
    }

    model_dir = get_model_dir(model_base_dir, stock_name)
    save_artifacts(
        model_dir,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        meta=meta,
    )

    return {
        "message": "trained_and_saved",
        "model_dir": str(model_dir),
        "meta": meta,
        "validation_metrics": {
            "rmse": rmse,
            "mae": mae,
        },
    }


def predict(
    *,
    stock_name: str,
    train_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    model_base_dir: str = ".",
    auto_train_if_missing: bool = True,
) -> Dict[str, Any]:
    """
    예측은 원칙적으로 저장된 모델을 로드해서 수행.
    - model이 없을 때만(옵션) 1회 학습 후 저장하고 재사용.
    """
    target_col = "target_close"

    feature_cols = _select_feature_cols(train_df)
    if len(feature_cols) == 0:
        return {"error": "사용 가능한 feature 컬럼이 없습니다."}

    train_df = train_df.sort_values("date").reset_index(drop=True)
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    train_df[target_col] = train_df[target_col].ffill()

    inference_df = inference_df.sort_values("date").reset_index(drop=True)
    inference_df[feature_cols] = inference_df[feature_cols].fillna(0)

    model_dir = get_model_dir(model_base_dir, stock_name)
    artifacts = load_artifacts(model_dir)

    if artifacts is None:
        if not auto_train_if_missing:
            return {"error": "저장된 모델이 없습니다. 먼저 /train으로 학습을 수행하세요."}
        train_result = train_and_save(stock_name=stock_name, train_df=train_df, model_base_dir=model_base_dir)
        if "error" in train_result:
            return train_result
        artifacts = load_artifacts(model_dir)

    if artifacts is None:
        return {"error": "모델 로드에 실패했습니다."}

    # feature 컬럼이 바뀌면 저장된 모델/스케일러와 불일치 → 안전하게 에러
    saved_cols = artifacts.meta.get("feature_cols", [])
    if list(saved_cols) != list(feature_cols):
        return {
            "error": "저장된 모델의 feature 컬럼과 현재 feature 컬럼이 다릅니다. 재학습이 필요합니다.",
            "saved_feature_cols": saved_cols,
            "current_feature_cols": feature_cols,
        }

    model = artifacts.model
    feature_scaler = artifacts.feature_scaler
    target_scaler = artifacts.target_scaler

    # -----------------------------
    # 백테스트(검증 구간): 시계열 holdout(마지막 20%)
    # -----------------------------
    split_row = int(len(train_df) * 0.8)
    if split_row <= SEQ_LEN:
        split_row = SEQ_LEN + 1
    if split_row >= len(train_df) - 1:
        split_row = max(SEQ_LEN + 1, len(train_df) - 2)

    df_val = train_df.iloc[split_row:].reset_index(drop=True)

    validation_metrics = {"rmse": None, "mae": None}
    historical_predictions: List[Dict[str, Any]] = []

    if len(df_val) > SEQ_LEN:
        X_val_scaled_full = feature_scaler.transform(df_val[feature_cols])
        y_val_scaled_full = target_scaler.transform(df_val[[target_col]])
        X_val_seq, y_val_seq = _create_sequences(X_val_scaled_full, y_val_scaled_full, seq_len=SEQ_LEN)

        val_pred_scaled = model.predict(X_val_seq, verbose=0)
        val_pred = target_scaler.inverse_transform(val_pred_scaled).flatten()
        y_val_real = target_scaler.inverse_transform(y_val_seq).flatten()

        validation_metrics["rmse"] = float(np.sqrt(mean_squared_error(y_val_real, val_pred)))
        validation_metrics["mae"] = float(mean_absolute_error(y_val_real, val_pred))

        # df_val의 i번째 예측은 df_val의 (SEQ_LEN+i) row의 target_close(=다음날 종가)를 예측
        pred_target_dates = pd.to_datetime(df_val["date"]).iloc[SEQ_LEN:].reset_index(drop=True)
        for i in range(len(val_pred)):
            historical_predictions.append({
                "date": str(pred_target_dates.iloc[i].date()),
                "actual_target_close": float(df_val[target_col].iloc[SEQ_LEN + i]),
                "predicted_target_close": float(val_pred[i]),
            })

    # -----------------------------
    # 최신 row로 다음 거래일 종가 예측
    # -----------------------------
    if len(inference_df) < SEQ_LEN:
        return {"error": f"추론 데이터가 너무 적습니다. 최소 {SEQ_LEN}개 이상 필요합니다."}

    X_infer_scaled = feature_scaler.transform(inference_df[feature_cols])
    latest_seq = X_infer_scaled[-SEQ_LEN:]
    latest_seq = np.expand_dims(latest_seq, axis=0)

    next_close_scaled = model.predict(latest_seq, verbose=0)
    next_close_pred = float(target_scaler.inverse_transform(next_close_scaled)[0][0])

    latest_close = float(inference_df["close"].iloc[-1])
    predicted_change = next_close_pred - latest_close
    predicted_change_pct = (predicted_change / latest_close) * 100 if latest_close != 0 else 0.0

    if next_close_pred > latest_close:
        direction = "UP"
    elif next_close_pred < latest_close:
        direction = "DOWN"
    else:
        direction = "SAME"

    return {
        "feature_cols": feature_cols,
        "sequence_length": SEQ_LEN,
        "model_meta": artifacts.meta,
        "validation_metrics": validation_metrics,
        "latest_input": {
            "last_date": str(pd.to_datetime(inference_df["date"].iloc[-1]).date()),
            "last_close": latest_close
        },
        "next_prediction": {
            "predicted_next_close": next_close_pred,
            "predicted_change": float(predicted_change),
            "predicted_change_pct": float(predicted_change_pct),
            "direction": direction
        },
        # 검증 구간(holdout) 예측을 중심으로 제공 (너무 길면 프론트가 무거워져서 30개로 제한)
        "historical_predictions": historical_predictions[-30:],
    }
