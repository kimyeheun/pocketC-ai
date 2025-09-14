from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from categorize.database.engine import make_engine_from_env
from categorize.utils import normalize_merchant

DEFAULT_OUTPUT = Path(os.getenv("MODEL_DIR", "./artifacts")).resolve()
DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)

def load_from_db(sample_limit: int | None) -> pd.DataFrame:
    """
    transactions + sub_categories를 이용해 라벨 데이터셋을 구성.
    - 텍스트: transactions.merchanr_name (정규화된 상호가 저장되어 있음)
    - 라벨  : sub_categories.sub_name
    """
    engine = make_engine_from_env()
    # 주의: 스키마 오타(merchanr_name)는 기존 코드/DB와 호환을 위해 그대로 사용한다.  :contentReference[oaicite:8]{index=8}
    base_sql = """
        SELECT t.merchanr_name AS text, s.sub_name AS label
        FROM transactions t
        JOIN sub_categories s ON s.sub_id = t.sub_id
        WHERE t.merchanr_name IS NOT NULL AND t.merchanr_name <> ''
    """
    if sample_limit:
        base_sql += f" LIMIT {int(sample_limit)}"
    df = pd.read_sql(base_sql, engine)
    # 혹시나 비정규화 데이터가 있다면 한 번 더 방어적 정규화
    df["text"] = df["text"].astype(str).map(normalize_merchant)
    df = df[(df["text"].str.len() > 0) & df["label"].notna()]
    return df

def load_from_csv(csv_path: str, text_col: str, label_col: str, sample_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if sample_limit:
        df = df.sample(n=min(sample_limit, len(df)), random_state=42)
    df["text"] = df[text_col].astype(str).map(normalize_merchant)
    df["label"] = df[label_col].astype(str)
    df = df[(df["text"].str.len() > 0) & df["label"].notna()]
    return df[["text", "label"]]

def train_and_save(df: pd.DataFrame, out_dir: Path, test_size: float = 0.15) -> Tuple[TfidfVectorizer, LogisticRegression, List[str]]:
    # 문자 n-gram 기반 TF-IDF (한글 상호명 특성상 char 2~5가 잘 동작)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2)
    X = vec.fit_transform(df["text"].tolist())

    labels: List[str] = sorted(df["label"].unique().tolist())
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y = np.array([label_to_idx[l] for l in df["label"].tolist()])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=2000, n_jobs=None)  # n_jobs는 일부 버전에서 미지원
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print("[Report] validation performance")
    print(classification_report(y_te, y_pred, target_names=labels, digits=4))

    # 저장
    (out_dir / "tfidf_vectorizer.joblib").write_bytes(joblib.dump(vec, None))
    (out_dir / "tfidf_model.joblib").write_bytes(joblib.dump(clf, None))
    (out_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved artifacts into: {out_dir}")
    return vec, clf, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["db", "csv"], default="db")
    ap.add_argument("--csv_path", type=str, help="CSV path (for --source csv)")
    ap.add_argument("--text_col", type=str, default="text", help="Text column name in CSV")
    ap.add_argument("--label_col", type=str, default="label", help="Label column name in CSV")
    ap.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick experiments")
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--test_size", type=float, default=0.15)
    args = ap.parse_args()

    if args.source == "db":
        df = load_from_db(sample_limit=args.limit)
    else:
        if not args.csv_path:
            raise ValueError("--csv_path is required when --source csv")
        df = load_from_csv(args.csv_path, args.text_col, args.label_col, args.limit)

    if len(df) < 50:
        print("[WARN] Not enough samples (<50). Results may be unstable.")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_and_save(df, out_dir, test_size=args.test_size)

if __name__ == "__main__":
    main()
