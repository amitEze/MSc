# src/data/build_base_table.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd

from .load_raw import (
    load_web_visits,
    load_app_usage,
    load_claims,
    load_churn_labels,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

HEALTH_KEYWORDS = [
    "nutrition", "diet", "mediterranean", "fiber",
    "weight", "weight loss", "bmi", "cholesterol", "lipid",
    "diabetes", "hypertension", "blood pressure",
    "exercise", "aerobic", "cardio", "strength training", "fitness",
    "sleep", "sleep hygiene", "restorative sleep", "sleep apnea",
    "stress", "mindfulness", "meditation", "wellbeing", "mental health",
]

ALLOWED_DOMAINS = ["wellco"]


def filter_web_visits(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    import re

    # build a safe regex pattern from keywords (escape special chars)
    escaped = [re.escape(k.lower()) for k in HEALTH_KEYWORDS]
    pattern = "(?:" + "|".join(escaped) + ")"

    # prepare searchable text for each column; if column missing, treat as empty
    title = df["title"].fillna("").astype(str).str.lower() if "title" in df.columns else pd.Series([""] * len(df), index=df.index)
    description = df["description"].fillna("").astype(str).str.lower() if "description" in df.columns else pd.Series([""] * len(df), index=df.index)

    # check each field for any keyword
    title_mask = title.str.contains(pattern, regex=True, na=False)
    desc_mask = description.str.contains(pattern, regex=True, na=False)

    if "url" in df.columns:
        url = df["url"].fillna("").astype(str).str.lower()
        url_mask = url.str.contains(pattern, regex=True, na=False)
        domain_mask = url.str.contains("|".join(ALLOWED_DOMAINS)) if ALLOWED_DOMAINS else pd.Series([False] * len(df), index=df.index)
    else:
        url_mask = pd.Series([False] * len(df), index=df.index)
        domain_mask = pd.Series([False] * len(df), index=df.index)

    # keep rows where any of url/title/description contain a keyword OR url matches allowed domains
    mask = title_mask | desc_mask | url_mask | domain_mask

    return df.loc[mask].copy()

def make_web_features(df: pd.DataFrame) -> pd.DataFrame:
    # compute unfiltered counts per member (before content filtering)
    if df.empty:
        return pd.DataFrame(columns=["member_id"])

    unfiltered_counts = df.groupby("member_id").size().rename("web_total_visits_unfiltered")

    df = filter_web_visits(df)
    if df.empty:
        return pd.DataFrame(columns=["member_id"])

    df = df.copy()
    df["member_id"] = df["member_id"].astype("int64")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    g = df.groupby("member_id")

    out = pd.DataFrame(index=g.size().index)
    out["web_total_visits"] = g.size().astype(float)

    if "url" in df.columns:
        out["web_unique_urls"] = g["url"].nunique().astype(float)
    else:
        out["web_unique_urls"] = 0.0

    if "timestamp" in df.columns:
        first_last = g["timestamp"].agg(["min", "max"])
        out["web_first_visit_ts"] = first_last["min"]
        out["web_last_visit_ts"] = first_last["max"]

        out["web_active_days"] = (
            g["timestamp"].apply(lambda x: x.dt.date.nunique()).astype(float)
        )

        global_max = df["timestamp"].max()
        out["web_days_since_last_visit"] = (
            (global_max - out["web_last_visit_ts"])
            .dt.total_seconds()
            .div(86400.0)
        )
    else:
        out["web_first_visit_ts"] = pd.NaT
        out["web_last_visit_ts"] = pd.NaT
        out["web_active_days"] = 0.0
        out["web_days_since_last_visit"] = np.nan

    out.reset_index(inplace=True)
    out.rename(columns={"index": "member_id"}, inplace=True)
    # attach unfiltered counts and compute relevant_url_rate = unfiltered / filtered
    out = out.join(unfiltered_counts, on="member_id", how="left")
    out["web_total_visits_unfiltered"] = out["web_total_visits_unfiltered"].fillna(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["relevant_url_rate"] = (
            out["web_total_visits_unfiltered"] / out["web_total_visits"].replace({0.0: np.nan})
        ).fillna(0.0)

    return out


def make_app_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["member_id"])

    df = df.copy()
    df["member_id"] = df["member_id"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    g = df.groupby("member_id")

    out = pd.DataFrame(index=g.size().index)
    out["app_sessions_total"] = g.size().astype(float)
    out["app_active_days"] = (
        g["timestamp"].apply(lambda x: x.dt.date.nunique()).astype(float)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        out["app_sessions_per_active_day"] = (
            out["app_sessions_total"] / out["app_active_days"].replace({0.0: np.nan})
        )

    first_last = g["timestamp"].agg(["min", "max"])
    out["app_first_session_ts"] = first_last["min"]
    out["app_last_session_ts"] = first_last["max"]

    global_max = df["timestamp"].max()
    out["app_days_since_last_session"] = (
        (global_max - out["app_last_session_ts"]).dt.total_seconds().div(86400.0)
    )

    out.reset_index(inplace=True)
    out.rename(columns={"index": "member_id"}, inplace=True)
    return out


def make_claims_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["member_id"])

    df = df.copy()
    df["member_id"] = df["member_id"].astype("int64")
    df["diagnosis_date"] = pd.to_datetime(df["diagnosis_date"], errors="coerce")

    g = df.groupby("member_id")

    out = pd.DataFrame(index=g.size().index)
    out["claims_total"] = g.size().astype(float)

    def has_icd(code: str, col_name: str) -> pd.Series:
        mask = df["icd_code"].astype(str).str.upper().eq(code.upper())
        return (
            df.loc[mask]
            .groupby("member_id")["icd_code"]
            .size()
            .gt(0)
            .astype(int)
            .rename(col_name)
        )

    out = out.join(has_icd("E11.9", "has_diabetes_E11_9"), how="left")
    out = out.join(has_icd("I10", "has_hypertension_I10"), how="left")
    out = out.join(has_icd("Z71.3", "has_diet_counseling_Z71_3"), how="left")

    for c in ["has_diabetes_E11_9", "has_hypertension_I10", "has_diet_counseling_Z71_3"]:
        out[c] = out[c].fillna(0).astype(int)

    first_last = g["diagnosis_date"].agg(["min", "max"])
    out["claims_first_diag_date"] = first_last["min"]
    out["claims_last_diag_date"] = first_last["max"]
    out["claims_span_days"] = (
        (out["claims_last_diag_date"] - out["claims_first_diag_date"])
        .dt.days.astype(float)
    )

    out.reset_index(inplace=True)
    out.rename(columns={"index": "member_id"}, inplace=True)
    return out


def make_labels_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["member_id"] = df["member_id"].astype("int64")

    if "churn" in df.columns:
        df["y_churn"] = df["churn"].astype(float)
    else:
        df["y_churn"] = np.nan

    if "outreach" in df.columns:
        df["t_outreach"] = df["outreach"].astype(float)
    else:
        df["t_outreach"] = np.nan

    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        min_signup = df["signup_date"].min()
        df["signup_days_since_min"] = (
            df["signup_date"] - min_signup
        ).dt.days.astype(float)
    else:
        df["signup_days_since_min"] = np.nan

    cols = [
        "member_id",
        "y_churn",
        "t_outreach",
        "signup_date",
        "signup_days_since_min",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def build_member_base_table(
    split: Literal["train", "test"],
    web_visits_path: Optional[str | Path] = None,
    app_usage_path: Optional[str | Path] = None,
    claims_path: Optional[str | Path] = None,
    churn_labels_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    if split == "train":
        web = load_web_visits(web_visits_path)
        app = load_app_usage(app_usage_path)
        claims = load_claims(claims_path)
        churn = load_churn_labels(churn_labels_path)
    elif split == "test":
        web = load_web_visits(web_visits_path or (RAW_DATA_DIR / "test_web_visits.csv"))
        app = load_app_usage(app_usage_path or (RAW_DATA_DIR / "test_app_usage.csv"))
        claims = load_claims(claims_path or (RAW_DATA_DIR / "test_claims.csv"))
        churn = load_churn_labels(
            churn_labels_path or (RAW_DATA_DIR / "test_churn_labels.csv")
        )
    else:
        raise ValueError("split must be 'train' or 'test'")

    labels = make_labels_features(churn)
    web_f = make_web_features(web)
    app_f = make_app_features(app)
    claims_f = make_claims_features(claims)

    base = labels
    for f in [web_f, app_f, claims_f]:
        if not f.empty:
            base = base.merge(f, on="member_id", how="left")

    count_cols = [
        c
        for c in base.columns
        if base[c].dtype.kind in "if"
        and c.startswith(("web_", "app_", "claims_"))
    ]
    for c in count_cols:
        base[c] = base[c].fillna(0.0)

    base = base.sort_values("member_id").reset_index(drop=True)
    return base


def save_member_base_table(
    split: Literal["train", "test"],
    **kwargs,
) -> Path:
    df = build_member_base_table(split=split, **kwargs)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / f"{split}_table.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {split} base table to {out_path} (rows={len(df)})")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build member-level base table.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--web_visits_path", type=str, default=None)
    parser.add_argument("--app_usage_path", type=str, default=None)
    parser.add_argument("--claims_path", type=str, default=None)
    parser.add_argument("--churn_labels_path", type=str, default=None)
    args = parser.parse_args()

    save_member_base_table(
        split=args.split,
        web_visits_path=args.web_visits_path,
        app_usage_path=args.app_usage_path,
        claims_path=args.claims_path,
        churn_labels_path=args.churn_labels_path,
    )
