#!/usr/bin/env python3
"""
PocketC Pipeline v2 — Excel → (Rules + Heuristics + Optional ML) → MySQL `transactions`

▶ 변화 요약 (v1 → v2)
- 하드코딩 룰 제거: DB 기반 `merchants / merchant_aliases / category_rules / user_overrides` 테이블 지원
- 퍼지 매칭(RapidFuzz) + 정규식 + 개인 오버라이드 우선순위 결합
- 저신뢰/미매핑 건 `curation_queue` 적재(운영자가 승인/수정)
- (옵션) LightGBM 분류기 학습·추론 지원: `python pocketc_pipeline.py train ...`
- 안전한 적재: 컬럼 오탈자(merchanr_name/status) 자동 감지, UPSERT idempotent
- 실행 모드: `ingest`, `train`, `init-db`, `dry-run` 지원

입력: 폴더 내 `[userID]_거래내역.xlsx`들 (각 파일 컬럼: '거래일시','적요','거래금액')
출력: `transactions` 테이블 (schema는 제공된 것과 호환)

필수 설치
  pip install pandas openpyxl SQLAlchemy pymysql python-dateutil pytz rapidfuzz lightgbm scikit-learn

환경변수
  DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME=pocketc
  MODEL_PATH=./pocketc_lgbm.bin (선택)
  CONFIRM_THRESHOLD=0.80  FUZZY_THRESHOLD=90  REGEX_ENABLE=1

사용 예
  # 1) 보조 테이블 생성(한 번)
  python pocketc_pipeline.py init-db
  # 2) 학습(선택)
  python pocketc_pipeline.py train --days 120
  # 3) 적재
  python pocketc_pipeline.py ingest /path/to/excel_folder --dry-run   # 시험
  python pocketc_pipeline.py ingest /path/to/excel_folder             # 실제 반영
"""
from __future__ import annotations
import os
import re
import sys
import zlib
import json
import time
import math
import hashlib
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List, Any

import pandas as pd
from dateutil import tz
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from rapidfuzz import fuzz, process as rf_process

# (옵션) ML
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    HAVE_LGBM = True
except Exception:
    HAVE_LGBM = False

KST = tz.gettz("Asia/Seoul")

# ==========================================
# DB: 보조 테이블 생성 (init-db)
# ==========================================
DDL_SUPPORTING = [
    # 표준화된 상호(고유) 목록
    """
    CREATE TABLE IF NOT EXISTS merchants (
      merchant_id   INT AUTO_INCREMENT PRIMARY KEY,
      normalized_name VARCHAR(200) NOT NULL,
      default_sub_id INT NULL,
      created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      UNIQUE KEY uq_merch_name (normalized_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    # 별칭/지점명 → merchants 매핑
    """
    CREATE TABLE IF NOT EXISTS merchant_aliases (
      alias_id    INT AUTO_INCREMENT PRIMARY KEY,
      alias_text  VARCHAR(200) NOT NULL,
      merchant_id INT NOT NULL,
      UNIQUE KEY uq_alias (alias_text),
      KEY idx_alias_mid (merchant_id),
      CONSTRAINT fk_alias_merch FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
        ON DELETE CASCADE ON UPDATE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    # 정규식 룰 → sub_id
    """
    CREATE TABLE IF NOT EXISTS category_rules (
      rule_id  INT AUTO_INCREMENT PRIMARY KEY,
      pattern  VARCHAR(300) NOT NULL,
      sub_id   INT NOT NULL,
      priority INT NOT NULL DEFAULT 100,
      enabled  TINYINT(1) NOT NULL DEFAULT 1,
      notes    VARCHAR(200) NULL,
      KEY idx_rule_pri (priority),
      KEY idx_rule_sub (sub_id),
      CONSTRAINT fk_rule_sub FOREIGN KEY (sub_id) REFERENCES sub_categories(sub_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    # 사용자 개별 오버라이드(가맹점 or 패턴)
    """
    CREATE TABLE IF NOT EXISTS user_overrides (
      override_id INT AUTO_INCREMENT PRIMARY KEY,
      user_id  INT NOT NULL,
      merchant_id INT NULL,
      pattern  VARCHAR(300) NULL,
      sub_id   INT NOT NULL,
      priority INT NOT NULL DEFAULT 10,
      enabled  TINYINT(1) NOT NULL DEFAULT 1,
      notes    VARCHAR(200) NULL,
      KEY idx_uo_user (user_id),
      KEY idx_uo_merch (merchant_id),
      CONSTRAINT fk_uo_sub FOREIGN KEY (sub_id) REFERENCES sub_categories(sub_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    # 저신뢰/미매핑 건 큐
    """
    CREATE TABLE IF NOT EXISTS curation_queue (
      curation_id INT AUTO_INCREMENT PRIMARY KEY,
      user_id INT NOT NULL,
      merchant_name VARCHAR(255) NULL,
      normalized_name VARCHAR(255) NULL,
      amount INT NULL,
      transacted_at DATETIME NULL,
      suggestion_sub_id INT NULL,
      suggestion_source VARCHAR(50) NULL,
      conf DECIMAL(5,2) NULL,
      reason VARCHAR(100) NULL,
      file VARCHAR(255) NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      KEY idx_cq_user (user_id),
      KEY idx_cq_time (transacted_at),
      KEY idx_cq_reason (reason)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
]

SEED_SQL = [
    # 가벼운 시드(원하면 추가 확장)
    """
    INSERT IGNORE INTO merchants(normalized_name, default_sub_id)
    VALUES
      ('스타벅스',''),('이디야',''),('투썸플레이스',''),('메가커피',''),('빽다방',''),
      ('배달의민족',''),('요기요',''),('쿠팡이츠',''),
      ('카카오T',''),('UT',''),('티맵대리',''),
      ('SOIL',''),('GS칼텍스',''),('SK에너지',''),('현대오일뱅크',''),
      ('SKT',''),('KT',''),('LGU+',''),
      ('넷플릭스',''),('디즈니+',''),('티빙',''),('왓챠',''),('유튜브프리미엄',''),('웨이브',''),
      ('쿠팡',''),('G마켓',''),('11번가',''),('SSG',''),('무신사',''),('올리브영','');
    """,
    # 카테고리 정규식 룰 시드(우선순위 높음)
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '(병원|의원|치과|한의원)', s.sub_id, 5, 1, '건강/의료-병원' FROM sub_categories s WHERE s.sub_name='병원' LIMIT 1;
    """,
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '약국', s.sub_id, 5, 1, '건강/의료-약국' FROM sub_categories s WHERE s.sub_name='약국' LIMIT 1;
    """,
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '(관리비|임대료|월세)', s.sub_id, 5, 1, '주거-월세/관리비' FROM sub_categories s WHERE s.sub_name='월세/관리비' LIMIT 1;
    """,
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '(전기요금|한국전력|한전)', s.sub_id, 5, 1, '주거-전기세' FROM sub_categories s WHERE s.sub_name='전기세' LIMIT 1;
    """,
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '(수도요금|수도사업본부)', s.sub_id, 5, 1, '주거-수도세' FROM sub_categories s WHERE s.sub_name='수도세' LIMIT 1;
    """,
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '(가스비|도시가스)', s.sub_id, 5, 1, '주거-가스비' FROM sub_categories s WHERE s.sub_name='가스비' LIMIT 1;
    """,
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '(티머니|T-money|지하철|도시철도|코레일)', s.sub_id, 5, 1, '교통-대중교통' FROM sub_categories s WHERE s.sub_name='대중교통' LIMIT 1;
    """,
    """
    INSERT IGNORE INTO category_rules(pattern, sub_id, priority, enabled, notes)
      SELECT '(국세청|지방세|세무서|건보공단|국민연금|고용보험|산재보험)', s.sub_id, 5, 1, '금융-세금/보험' FROM sub_categories s WHERE s.sub_name='세금/보험' LIMIT 1;
    """,
]

# ==========================================
# 유틸: 파싱/정규화
# ==========================================
BRACKET_RE = re.compile(r"[\[\(\{].*?[\]\)\}]")
WS_RE = re.compile(r"\s+")
NON_ALNUM_KO_EN = re.compile(r"[^0-9A-Za-z가-힣\s]", re.UNICODE)
BRANCH_SUFFIX_RE = re.compile(r"(역|점|센터|점포|지점|영업점|사옥|타워)\s*\w*$")


def parse_amount(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    s = s.replace(",", "").replace("원", "").replace("₩", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return int(float(s))
    except Exception:
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else 0


def parse_datetime_kst(x) -> datetime:
    # pandas가 datetime으로 읽으면 그대로 사용
    if isinstance(x, datetime):
        dt = x
    else:
        s = str(x).strip()
        # Excel serial number 지원
        if s.replace(".", "", 1).isdigit() and float(s) > 20000:
            origin = datetime(1899, 12, 30)  # Excel 기준
            dt = origin + timedelta(days=float(s))
        else:
            dt = None
            for fmt in [
                "%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%Y/%m/%d %H:%M:%S","%Y/%m/%d %H:%M",
                "%Y.%m.%d %H:%M","%Y%m%d%H%M%S","%Y%m%d%H%M","%Y-%m-%d","%Y/%m/%d",
            ]:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except Exception:
                    pass
            if dt is None:
                raise ValueError(f"Unrecognized datetime format: {x}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt


def normalize_merchant(raw: Any) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw).replace("\u200b", "")
    s = BRACKET_RE.sub(" ", s)
    s = NON_ALNUM_KO_EN.sub(" ", s)
    s = BRANCH_SUFFIX_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip()
    return s

# ==========================================
# 구조체
# ==========================================
@dataclass
class SubRow:
    sub_id: int
    major_id: int
    sub_name: str

@dataclass
class CategoryHit:
    sub_name: str
    source: str
    conf: float

# ==========================================
# DB 접근
# ==========================================

def make_engine_from_env() -> Engine:
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    name = os.getenv("DB_NAME", "pocketc")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True, future=True)


def ensure_supporting_tables(engine: Engine):
    with engine.begin() as conn:
        for ddl in DDL_SUPPORTING:
            conn.execute(text(ddl))
        for seed in SEED_SQL:
            conn.execute(text(seed))


def load_sub_map(engine: Engine) -> Dict[str, SubRow]:
    df = pd.read_sql("SELECT sub_id, major_id, sub_name FROM sub_categories", engine)
    mapping: Dict[str, SubRow] = {}
    for _, r in df.iterrows():
        mapping[str(r.sub_name)] = SubRow(int(r.sub_id), int(r.major_id), str(r.sub_name))
    return mapping


def load_merchants_and_aliases(engine: Engine):
    merch_df = pd.read_sql("SELECT merchant_id, normalized_name, default_sub_id FROM merchants", engine)
    alias_df = pd.read_sql("SELECT alias_text, merchant_id FROM merchant_aliases", engine)
    # alias_text → merchant_id
    alias2mid = {str(a.alias_text): int(a.merchant_id) for _, a in alias_df.iterrows()}
    # normalized_name → merchant_id
    name2mid = {str(m.normalized_name): int(m.merchant_id) for _, m in merch_df.iterrows()}
    mid2default = {int(m.merchant_id): (int(m.default_sub_id) if pd.notna(m.default_sub_id) else None)
                   for _, m in merch_df.iterrows()}
    return name2mid, alias2mid, mid2default


def load_rules(engine: Engine):
    df = pd.read_sql("SELECT pattern, sub_id, priority FROM category_rules WHERE enabled=1 ORDER BY priority ASC", engine)
    rules: List[Tuple[re.Pattern, int, int]] = []
    for _, r in df.iterrows():
        try:
            rules.append((re.compile(r.pattern), int(r.sub_id), int(r.priority)))
        except re.error:
            print(f"[WARN] bad regex skipped: {r.pattern}")
    return rules


def load_user_overrides(engine: Engine) -> List[dict]:
    df = pd.read_sql("SELECT override_id, user_id, merchant_id, pattern, sub_id, priority FROM user_overrides WHERE enabled=1 ORDER BY priority ASC", engine)
    return df.to_dict(orient='records')


def detect_transactions_columns(engine: Engine):
    sql = text("""
      SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME='transactions'
    """)
    with engine.begin() as conn:
        cols = {r[0] for r in conn.execute(sql)}
    # 오탈자 대응
    merchant_col = 'merchant_name' if 'merchant_name' in cols else 'merchanr_name'
    status_col = 'status' if 'status' in cols else 'staus'
    return merchant_col, status_col

# ==========================================
# 분류기
# ==========================================

def classify_row(merchant_norm: str, amount: int, kst_dt: datetime,
                 user_id: int,
                 name2mid: Dict[str,int], alias2mid: Dict[str,int], mid2default: Dict[int, Optional[int]],
                 rules: List[Tuple[re.Pattern,int,int]], overrides: List[dict],
                 sub_map: Dict[str, SubRow],
                 model: Any = None, label_encoder: Dict[int,int] = None,
                 fuzzy_threshold: int = 90, confirm_th: float = 0.8,
                 regex_enable: bool = True) -> CategoryHit:
    # 0) 사용자 오버라이드 (가맹점/패턴)
    for o in overrides:
        if o['user_id'] != user_id:
            continue
        if o['merchant_id'] is not None:
            mid = int(o['merchant_id'])
            if merchant_norm in name2mid and name2mid[merchant_norm] == mid:
                sub_id = int(o['sub_id'])
                sub_name = _sub_name_by_id(sub_map, sub_id)
                return CategoryHit(sub_name, 'override:merchant', 0.98)
        if o['pattern']:
            try:
                if re.search(o['pattern'], merchant_norm):
                    sub_id = int(o['sub_id'])
                    sub_name = _sub_name_by_id(sub_map, sub_id)
                    return CategoryHit(sub_name, 'override:pattern', 0.96)
            except re.error:
                pass

    # 1) 정확 일치: alias → merchant
    if merchant_norm in alias2mid:
        mid = alias2mid[merchant_norm]
        sub_id = mid2default.get(mid)
        if sub_id:
            sub_name = _sub_name_by_id(sub_map, sub_id)
            return CategoryHit(sub_name, 'alias_map', 0.95)
        # default_sub_id 미지정 시 규칙/휴리스틱으로 계속 진행

    # 2) 정확 일치: normalized_name → merchant
    if merchant_norm in name2mid:
        mid = name2mid[merchant_norm]
        sub_id = mid2default.get(mid)
        if sub_id:
            sub_name = _sub_name_by_id(sub_map, sub_id)
            return CategoryHit(sub_name, 'merchant_map', 0.95)

    # 3) 퍼지 매칭(merchant catalog)
    if name2mid:
        choice = rf_process.extractOne(merchant_norm, list(name2mid.keys()), scorer=fuzz.token_sort_ratio)
        if choice and choice[1] >= fuzzy_threshold:
            mid = name2mid[choice[0]]
            sub_id = mid2default.get(mid)
            if sub_id:
                sub_name = _sub_name_by_id(sub_map, sub_id)
                return CategoryHit(sub_name, f'fuzzy@{choice[1]}', 0.9)

    # 4) 정규식 룰
    if regex_enable:
        for pat, sub_id, _pri in rules:
            if pat.search(merchant_norm):
                sub_name = _sub_name_by_id(sub_map, sub_id)
                return CategoryHit(sub_name, 'regex', 0.9)

    # 5) 휴리스틱(식비 부분)
    hour = kst_dt.hour
    if re.search(r"(카페|커피|COFFEE|CAFE)", merchant_norm) and (1000 <= amount <= 15000):
        return CategoryHit("커피", "heuristic:cafe", 0.8)
    if re.search(r"(편의점|CU|GS25|세븐일레븐|이마트24)", merchant_norm) and amount <= 20000:
        return CategoryHit("간식", "heuristic:conv_store", 0.75)
    if ("마트" in merchant_norm or "식당" in merchant_norm):
        if 6000 <= amount <= 50000 and (11 <= hour <= 14 or 17 <= hour <= 21):
            return CategoryHit("외식", "heuristic:dining", 0.65)
        if amount > 20000:
            return CategoryHit("식재료", "heuristic:grocery", 0.6)

    # 6) (옵션) ML 추론
    if model is not None:
        feats = _featurize(merchant_norm, amount, kst_dt)
        proba = model.predict(feats)[0]
        # label_encoder: sub_id -> index, index -> sub_name
        pred_idx = int(proba.argmax())
        conf = float(proba[pred_idx])
        sub_id = _index_to_sub_id(label_encoder, pred_idx)
        sub_name = _sub_name_by_id(sub_map, sub_id) if sub_id else "기타"
        source = 'ml' if conf >= confirm_th else 'ml-weak'
        return CategoryHit(sub_name, source, conf)

    return CategoryHit("기타", "fallback", 0.2)


# ==========================================
# ML 보조 함수 (간단 피처)
# ==========================================
TOK_SPLIT = re.compile(r"[\s/\-_,.]+")


def _featurize(merchant_norm: str, amount: int, kst_dt: datetime):
    # 아주 간단한 bag-of-words + 수치 피처 (실전에서는 TF-IDF 추천)
    tokens = [t for t in TOK_SPLIT.split(merchant_norm) if t]
    vocab = [
        '스타벅스','이디야','투썸','메가커피','빽다방','배달','요기요','쿠팡이츠','카카오T','택시','주유','GS칼텍스',
        'SOIL','전기','수도','가스','넷플릭스','티빙','디즈니','쿠팡','G마켓','11번가','SSG','무신사','올리브영',
        '병원','약국','영화','게임','마트','식당','편의점','카페'
    ]
    fv = [1.0 if any(v in tokens or v in merchant_norm for v in [w]) else 0.0 for w in vocab]
    hour = kst_dt.hour
    dow = (kst_dt.weekday()+1)  # 1=Mon
    num_feats = [amount, hour, dow]
    import numpy as np
    arr = np.array([fv + num_feats])
    return arr


def _build_label_encoder(engine: Engine) -> Tuple[Dict[int,int], Dict[int,int]]:
    # sub_id ↔ index
    df = pd.read_sql("SELECT sub_id FROM sub_categories ORDER BY sub_id", engine)
    sub_ids = [int(x) for x in df.sub_id.tolist()]
    idx_map = {sid:i for i, sid in enumerate(sub_ids)}
    inv_map = {i:sid for i, sid in enumerate(sub_ids)}
    return idx_map, inv_map


def _index_to_sub_id(inv_map: Dict[int,int], idx: int) -> Optional[int]:
    return inv_map.get(idx)


def _sub_name_by_id(sub_map: Dict[str, SubRow], sub_id: int) -> str:
    for name, row in sub_map.items():
        if row.sub_id == sub_id:
            return name
    return "기타"

# ==========================================
# 트랜잭션 적재
# ==========================================
INSERT_SQL_TPL = """
    INSERT INTO transactions (
        transaction_id, user_id, sub_id, major_id, transacted_at,
        amount, {merchant_col}, {status_col}, created_at, updated_at
    ) VALUES (
        :transaction_id, :user_id, :sub_id, :major_id, :transacted_at,
        :amount, :merchant_name, :status, :created_at, :updated_at
    )
    ON DUPLICATE KEY UPDATE
        amount = VALUES(amount),
        {merchant_col} = VALUES({merchant_col}),
        {status_col} = VALUES({status_col}),
        updated_at = VALUES(updated_at)
"""


def make_txn_id(user_id: int, ts: datetime, amount: int, merchant: str) -> int:
    key = f"{user_id}|{int(ts.timestamp())}|{amount}|{merchant}".encode("utf-8")
    crc = zlib.crc32(key) & 0xFFFFFFFF
    return int(crc % 2000000000)


# ==========================================
# 파일 처리
# ==========================================

def process_file(path: str, engine: Engine, sub_map: Dict[str, SubRow],
                 name2mid, alias2mid, mid2default, rules, overrides,
                 merchant_col: str, status_col: str,
                 unknown_log: List[dict],
                 model=None, label_inv=None,
                 dry_run=False,
                 confirm_th=0.8, fuzzy_th=90, regex_enable=True):
    m = re.search(r"(\d+)_거래내역\.xlsx$", os.path.basename(path))
    if not m:
        print(f"[SKIP] filename pattern mismatch: {path}")
        return
    user_id = int(m.group(1))

    df = pd.read_excel(path)
    need = ['거래일시','적요','거래금액']
    for n in need:
        if n not in df.columns:
            raise KeyError(f"Column '{n}' not found in {path}. Found: {list(df.columns)}")

    now_kst = datetime.now(KST)
    ins_rows = []

    for _, row in df.iterrows():
        try:
            kst_dt = parse_datetime_kst(row['거래일시'])
        except Exception:
            unknown_log.append({'file': path,'user_id': user_id,'reason':'bad_datetime','거래일시': str(row['거래일시']), '적요': str(row['적요'])})
            continue
        merchant_raw = row['적요']
        merchant_norm = normalize_merchant(merchant_raw)
        amount = parse_amount(row['거래금액'])

        hit = classify_row(
            merchant_norm, amount, kst_dt, user_id,
            name2mid, alias2mid, mid2default, rules, overrides,
            sub_map, model, label_inv,
            fuzzy_threshold=fuzzy_th, confirm_th=confirm_th, regex_enable=regex_enable
        )

        sub_name = hit.sub_name if hit.sub_name in sub_map else '기타'
        sub = sub_map[sub_name]
        status = '반영' if hit.conf >= confirm_th and hit.source not in ('ml-weak','fallback') else '미반영'

        txn_id = make_txn_id(user_id, kst_dt, amount, merchant_norm)
        ins_rows.append({
            'transaction_id': txn_id,
            'user_id': user_id,
            'sub_id': sub.sub_id,
            'major_id': sub.major_id,
            'transacted_at': kst_dt.astimezone(KST).replace(tzinfo=None),
            'amount': int(amount),
            'merchant_name': merchant_norm,
            'status': status,
            'created_at': now_kst.replace(tzinfo=None),
            'updated_at': now_kst.replace(tzinfo=None),
        })

        # 저신뢰 큐 적재
        if status == '미반영':
            unknown_log.append({
                'user_id': user_id,
                'merchant_name': merchant_raw,
                'normalized_name': merchant_norm,
                'amount': int(amount),
                'transacted_at': kst_dt.astimezone(KST).replace(tzinfo=None),
                'suggestion_sub_id': sub.sub_id,
                'suggestion_source': hit.source,
                'conf': round(hit.conf, 3),
                'reason': 'low_conf',
                'file': path,
            })

    if not ins_rows:
        return

    if dry_run:
        print(f"[DRY] Would insert {len(ins_rows)} rows from {os.path.basename(path)}")
        return

    insert_sql = text(INSERT_SQL_TPL.format(merchant_col=merchant_col, status_col=status_col))
    with engine.begin() as conn:
        conn.execute(insert_sql, ins_rows)


# ==========================================
# ML 학습 (선택)
# ==========================================

def train_model(engine: Engine, days: int, model_path: str):
    if not HAVE_LGBM:
        print("[WARN] lightgbm not installed. Skipping train.")
        return
    # 이미 반영된 데이터만 학습
    q = text(
        """
        SELECT t.transacted_at, t.amount, t.user_id, t.{merchant_col} AS merchant_name, t.sub_id
        FROM transactions t
        WHERE t.{status_col}='반영' AND t.transacted_at >= :since
        """.format(*(), merchant_col=detect_transactions_columns(engine)[0], status_col=detect_transactions_columns(engine)[1])
    )
    since = datetime.now(KST) - timedelta(days=days)
    df = pd.read_sql(q, engine, params={'since': since.replace(tzinfo=None)})
    if df.empty:
        print("[WARN] No data to train.")
        return

    idx_map, inv_map = _build_label_encoder(engine)

    # 피처/라벨 생성
    X = []
    y = []
    for _, r in df.iterrows():
        dt = r['transacted_at'] if isinstance(r['transacted_at'], datetime) else parse_datetime_kst(r['transacted_at'])
        X.append(_featurize(normalize_merchant(r['merchant_name']), int(r['amount']), dt)[0])
        y.append(idx_map.get(int(r['sub_id']), 0))

    import numpy as np
    X = np.array(X)
    y = np.array(y)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_data = lgb.Dataset(Xtr, label=ytr)
    valid_data = lgb.Dataset(Xte, label=yte, reference=train_data)

    params = dict(objective='multiclass', num_class=len(inv_map), learning_rate=0.1, num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1)
    model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=200, early_stopping_rounds=20, verbose_eval=False)

    # 간단 평가지표
    ypred = model.predict(Xte)
    yhat = ypred.argmax(axis=1)
    f1 = f1_score(yte, yhat, average='macro')
    print(f"[INFO] Trained LightGBM macro-F1={f1:.3f}, saving to {model_path}")

    model.save_model(model_path)


# ==========================================
# curation_queue 저장
# ==========================================

def flush_curation_queue(engine: Engine, rows: List[dict]):
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(text(
            """
            INSERT INTO curation_queue
              (user_id, merchant_name, normalized_name, amount, transacted_at, suggestion_sub_id, suggestion_source, conf, reason, file)
            VALUES
              (:user_id, :merchant_name, :normalized_name, :amount, :transacted_at, :suggestion_sub_id, :suggestion_source, :conf, :reason, :file)
            """
        ), rows)


# ==========================================
# 엔트리포인트
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_init = sub.add_parser('init-db', help='보조 테이블 생성 및 시드')

    ap_ing = sub.add_parser('ingest', help='엑셀 폴더 적재')
    ap_ing.add_argument('folder', help='[userID]_거래내역.xlsx 들이 있는 폴더')
    ap_ing.add_argument('--dry-run', action='store_true')
    ap_ing.add_argument('--fuzzy', type=int, default=int(os.getenv('FUZZY_THRESHOLD', '90')))
    ap_ing.add_argument('--confirm', type=float, default=float(os.getenv('CONFIRM_THRESHOLD', '0.8')))
    ap_ing.add_argument('--no-regex', action='store_true')

    ap_tr = sub.add_parser('train', help='LightGBM 학습(선택)')
    ap_tr.add_argument('--days', type=int, default=120)
    ap_tr.add_argument('--model', type=str, default=os.getenv('MODEL_PATH', './pocketc_lgbm.bin'))

    args = ap.parse_args()

    engine = make_engine_from_env()

    if args.cmd == 'init-db':
        ensure_supporting_tables(engine)
        print('[OK] Supporting tables created & seeded.')
        return

    if args.cmd == 'train':
        train_model(engine, args.days, args.model)
        return

    # ingest
    folder = args.folder
    sub_map = load_sub_map(engine)
    name2mid, alias2mid, mid2default = load_merchants_and_aliases(engine)
    rules = load_rules(engine)
    overrides = load_user_overrides(engine)
    merchant_col, status_col = detect_transactions_columns(engine)

    # (옵션) 모델 로드
    model = None
    label_inv = None
    model_path = os.getenv('MODEL_PATH', './pocketc_lgbm.bin')
    if os.path.exists(model_path) and HAVE_LGBM:
        model = lgb.Booster(model_file=model_path)
        # inv_map
        _idx, label_inv = _build_label_encoder(engine)
        print(f"[INFO] Loaded model: {model_path}")

    excel_paths = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith('_거래내역.xlsx')
    ]
    excel_paths.sort()

    unknown_log: List[dict] = []

    for p in excel_paths:
        print(f"[INFO] Processing {p}")
        try:
            process_file(
                p, engine, sub_map,
                name2mid, alias2mid, mid2default,
                rules, overrides,
                merchant_col, status_col,
                unknown_log,
                model=model, label_inv=label_inv,
                dry_run=args.dry_run,
                confirm_th=args.confirm,
                fuzzy_th=args.fuzzy,
                regex_enable=not args.no_regex,
            )
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            unknown_log.append({'file': p, 'reason': 'exception', 'detail': str(e)})

    # 큐 적재
    if unknown_log:
        if args.dry_run:
            print(f"[DRY] {len(unknown_log)} rows would be inserted into curation_queue")
        else:
            flush_curation_queue(engine, unknown_log)
            print(f"[INFO] Inserted {len(unknown_log)} rows into curation_queue")
    else:
        print('[INFO] No low-confidence items.')


if __name__ == '__main__':
    main()
