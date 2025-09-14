from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from category_lib import CategoryHit, REGEX_RULES, CAFE_HINT, CONVENIENCE_HINT
from database import SubRow
from utils import parse_amount, parse_datetime_kst, normalize_merchant, KST

# (선택) 외부 모델 의존성: 존재하면 사용, 없으면 안전히 건너뛴다.
# 원본 코드에서는 model.main 에 TFIDF_VEC/CLF, KOBERT_* 및 init_ml_once 가 있다고 가정함. :contentReference[oaicite:6]{index=6}
try:
    from model.main import init_ml_once, TFIDF_VEC, TFIDF_CLF, KOBERT_TOK, KOBERT_MODEL, KOBERT_CLF, kobert_embed, ML_CONFIRM
except Exception:
    def init_ml_once():
        return
    TFIDF_VEC = TFIDF_CLF = None
    KOBERT_TOK = KOBERT_MODEL = KOBERT_CLF = None
    def kobert_embed(x):  # pragma: no cover
        return None
    ML_CONFIRM = 0.75


# --------------------------
# Rule 체인 (모듈화의 핵심)
# --------------------------
class Rule:
    def apply(self, merchant: str, amount: int, kst_hour: int) -> CategoryHit | None:
        raise NotImplementedError


class RegexRule(Rule):
    def __init__(self, patterns: List[Tuple[re.Pattern, str]]):
        self.patterns = patterns

    def apply(self, merchant: str, amount: int, kst_hour: int) -> CategoryHit | None:
        for pat, sub_name in self.patterns:
            if pat.search(merchant):
                return CategoryHit(sub_name, "regex")
        return None


class CafeRule(Rule):
    def apply(self, merchant: str, amount: int, kst_hour: int) -> CategoryHit | None:
        if CAFE_HINT.search(merchant):
            return CategoryHit("커피", "heuristic:cafe")
        return None


class ConvenienceRule(Rule):
    def apply(self, merchant: str, amount: int, kst_hour: int) -> CategoryHit | None:
        if CONVENIENCE_HINT.search(merchant):
            return CategoryHit("간식", "heuristic:conv_store")
        return None


class DiningTimeRule(Rule):
    def apply(self, merchant: str, amount: int, kst_hour: int) -> CategoryHit | None:
        if (11 <= kst_hour <= 14 or 18 <= kst_hour <= 21) and amount > 10_000:
            return CategoryHit("외식", "heuristic:dining")
        return None


class MLRule(Rule):
    """
    ML 보조 판단: 확신(confidence)이 높으면 라벨을 제안.
    주의: 라벨 인덱스 ↔ sub_name 매핑 아티팩트가 없다면, 보수적으로 '기타'로 둔다. (운영 임계치 ML_CONFIRM 사용)
    BUGFIX: 원본의 ('기타' 'ml:tfidf') 콤마 누락과 conf 미반영 문제 수정. :contentReference[oaicite:7]{index=7}
    """
    def __init__(self, label_inverse_map: List[str] | None = None):
        # label_inverse_map 이 있으면 해당 값을 sub_name 으로 사용
        self.label_inverse_map = label_inverse_map

    def apply(self, merchant: str, amount: int, kst_hour: int) -> CategoryHit | None:
        init_ml_once()
        candidates: List[Tuple[str, str, float]] = []  # (sub_name, source, score)

        # TF-IDF 계열
        if TFIDF_VEC is not None and TFIDF_CLF is not None:
            try:
                X = TFIDF_VEC.transform([merchant])
                if hasattr(TFIDF_CLF, "predict_proba"):
                    p = TFIDF_CLF.predict_proba(X)[0]
                else:
                    d = TFIDF_CLF.decision_function(X)[0]
                    d = d - d.max()
                    p = np.exp(d) / np.exp(d).sum()
                idx = int(np.argmax(p))
                conf = float(p[idx])
                if conf >= ML_CONFIRM:
                    sub_name = (
                        self.label_inverse_map[idx]
                        if self.label_inverse_map is not None and 0 <= idx < len(self.label_inverse_map)
                        else "기타"
                    )
                    candidates.append((sub_name, "ml:tfidf", conf))
            except Exception:
                pass

        # KoBERT 계열
        if KOBERT_TOK is not None and KOBERT_MODEL is not None and KOBERT_CLF is not None:
            try:
                emb = kobert_embed([merchant])
                p = KOBERT_CLF.predict_proba(emb)[0]
                idx = int(np.argmax(p))
                conf = float(p[idx])
                if conf >= ML_CONFIRM:
                    sub_name = (
                        self.label_inverse_map[idx]
                        if self.label_inverse_map is not None and 0 <= idx < len(self.label_inverse_map)
                        else "기타"
                    )
                    candidates.append((sub_name, "ml:kobert", conf))
            except Exception:
                pass

        if candidates:
            # score 최대인 후보 선택
            sub_name, source, _score = max(candidates, key=lambda x: x[2])
            return CategoryHit(sub_name, source)

        return None


class Categorizer:
    def __init__(self, rules: List[Rule], fallback: str = "기타"):
        self.rules = rules
        self.fallback = fallback

    def classify(self, merchant: str, amount: int, kst_dt: datetime) -> CategoryHit:
        hour = kst_dt.hour
        for r in self.rules:
            hit = r.apply(merchant, amount, hour)
            if hit:
                return hit
        return CategoryHit(self.fallback, "fallback")


# 기본 체인 구성
DEFAULT_CHAIN = Categorizer(
    rules=[
        RegexRule(REGEX_RULES),
        CafeRule(),
        ConvenienceRule(),
        DiningTimeRule(),
        MLRule(label_inverse_map=None),  # 아티팩트 없으면 None
    ],
    fallback="기타",
)


def classify_row(merchant: str, amount: int, kst_dt: datetime) -> CategoryHit:
    """
    호환용 함수 (기존 시그니처 유지). 내부는 Rule 체인으로 모듈화. :contentReference[oaicite:8]{index=8}
    """
    return DEFAULT_CHAIN.classify(merchant, amount, kst_dt)


# --------------------------
# 파일 처리 및 DB 반영
# --------------------------
INSERT_SQL = text(
    """
    INSERT INTO transactions (
        user_id, sub_id, major_id, transacted_at,
        amount, merchanr_name, staus, created_at, updated_at
    ) VALUES (
        :user_id, :sub_id, :major_id, :transacted_at,
        :amount, :merchanr_name, :staus, :created_at, :updated_at
    )
    ON DUPLICATE KEY UPDATE
        amount = VALUES(amount),
        merchanr_name = VALUES(merchanr_name),
        staus = VALUES(staus),
        updated_at = VALUES(updated_at)
    """
)


def process_file(path: str, engine: Engine, sub_map: Dict[str, SubRow], unknown_log: List[dict]):
    m = re.search(r"(\d+)_거래내역\.xlsx$", os.path.basename(path))
    if not m:
        print(f"[SKIP] file name not matching pattern: {path}")
        return
    user_id = int(m.group(1))

    df = pd.read_excel(path, dtype=str)
    cols = {c.strip(): c for c in df.columns}
    need = ["거래일시", "적요", "출금액"]
    for n in need:
        if n not in cols:
            raise KeyError(f"Column '{n}' not found in {path}. Found: {list(df.columns)}")

    now_kst = datetime.now(KST)

    rows = []
    for _, row in df.iterrows():
        raw_dt = row[cols["거래일시"]]
        raw_nm = row[cols["적요"]]
        raw_amt = row[cols["출금액"]]

        try:
            kst_dt = parse_datetime_kst(raw_dt)
        except Exception:
            unknown_log.append(
                {
                    "user_id": user_id,
                    "reason": "bad_datetime",
                    "거래일시": str(raw_dt),
                    "적요": str(raw_nm),
                    "출금액": str(raw_amt),
                    "file": path,
                }
            )
            continue

        merchant = normalize_merchant(raw_nm)
        amount = parse_amount(raw_amt)

        hit = classify_row(merchant, amount, kst_dt)
        sub_name = hit.sub_name

        # map sub_name -> (sub_id, major_id); fallback to 기타
        if sub_name not in sub_map:
            unknown_log.append(
                {
                    "user_id": user_id,
                    "reason": "unknown_sub_name",
                    "sub_name": sub_name,
                    "merchant": merchant,
                    "amount": amount,
                    "file": path,
                }
            )
            sub_name = "기타" if "기타" in sub_map else list(sub_map.keys())[0]

        sub = sub_map[sub_name]
        status = "미반영"

        rows.append(
            {
                "user_id": user_id,
                "sub_id": sub.sub_id,
                "major_id": sub.major_id,
                "transacted_at": kst_dt.astimezone(KST).replace(tzinfo=None),  # naive KST
                "amount": amount,
                "merchanr_name": merchant,
                "staus": status,
                "created_at": now_kst.replace(tzinfo=None),
                "updated_at": now_kst.replace(tzinfo=None),
            }
        )

    if not rows:
        return

    with engine.begin() as conn:
        conn.execute(INSERT_SQL, rows)
