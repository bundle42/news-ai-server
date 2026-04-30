import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tensorflow import keras


def _safe_slug(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "unknown"
    return "".join(ch if ch.isalnum() else "_" for ch in name)


@dataclass(frozen=True)
class ModelArtifacts:
    model: keras.Model
    feature_scaler: Any
    target_scaler: Any
    meta: Dict[str, Any]


def get_model_dir(base_dir: str, stock_name: str) -> Path:
    return Path(base_dir).resolve() / "models" / _safe_slug(stock_name)


def load_artifacts(model_dir: Path) -> Optional[ModelArtifacts]:
    meta_path = model_dir / "meta.json"
    model_path = model_dir / "keras_model.keras"
    feature_scaler_path = model_dir / "feature_scaler.pkl"
    target_scaler_path = model_dir / "target_scaler.pkl"

    if not (meta_path.exists() and model_path.exists() and feature_scaler_path.exists() and target_scaler_path.exists()):
        return None

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model = keras.models.load_model(model_path)
    with feature_scaler_path.open("rb") as f:
        feature_scaler = pickle.load(f)
    with target_scaler_path.open("rb") as f:
        target_scaler = pickle.load(f)

    return ModelArtifacts(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        meta=meta,
    )


def save_artifacts(
    model_dir: Path,
    *,
    model: keras.Model,
    feature_scaler: Any,
    target_scaler: Any,
    meta: Dict[str, Any],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    meta_path = model_dir / "meta.json"
    model_path = model_dir / "keras_model.keras"
    feature_scaler_path = model_dir / "feature_scaler.pkl"
    target_scaler_path = model_dir / "target_scaler.pkl"

    tmp_meta = meta_path.with_suffix(".json.tmp")
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp_meta, meta_path)

    # Keras save is atomic-ish (writes temp files); still keep consistent names.
    model.save(model_path)
    with feature_scaler_path.open("wb") as f:
        pickle.dump(feature_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    with target_scaler_path.open("wb") as f:
        pickle.dump(target_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

