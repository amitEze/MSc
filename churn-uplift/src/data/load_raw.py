# src/data/load_raw.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"


def _resolve(path: Optional[str | Path], default_name: str) -> Path:
    return Path(path) if path is not None else RAW_DATA_DIR / default_name


def load_web_visits(path: Optional[str | Path] = None) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path, "web_visits.csv"))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def load_app_usage(path: Optional[str | Path] = None) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path, "app_usage.csv"))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def load_claims(path: Optional[str | Path] = None) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path, "claims.csv"))
    if "diagnosis_date" in df.columns:
        df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors="coerce")
    return df


def load_churn_labels(path: Optional[str | Path] = None) -> pd.DataFrame:
    df = pd.read_csv(_resolve(path, "churn_labels.csv"))
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    return df
