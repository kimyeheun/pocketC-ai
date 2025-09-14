from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def make_engine_from_env() -> Engine:
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    # 특수문자 안전
    password = quote_plus(os.getenv("DB_PASSWORD", "yen1212"))
    name = os.getenv("DB_NAME", "pocketc")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
    engine = create_engine(url, pool_pre_ping=True, future=True)
    # 연결 즉시 검증
    with engine.connect() as conn:
        conn.exec_driver_sql("SELECT 1")
    return engine

@dataclass
class TransactionRepository:
    engine: Engine

    def fetch_last_days(self, days: int = 30, user_id: Optional[int] = None) -> pd.DataFrame:
        """
        최근 N일 거래내역 로드.
        """
        cutoff = datetime.now() - timedelta(days=days)
        base_sql = """
SELECT 
    t.transaction_id,
    t.user_id,
    COALESCE(s.sub_name, '기타')   AS sub_id,     -- ← 누락시 기본값
    COALESCE(m.major_name, '기타') AS major_id,   -- ← 누락시 기본값
    t.transacted_at,
    t.amount,
    t.merchanr_name,
    t.staus
FROM transactions AS t
LEFT JOIN sub_categories  AS s ON t.sub_id   = s.sub_id
LEFT JOIN major_categories AS m ON s.major_id = m.major_id
        """
        params = {"cutoff": cutoff}
        if user_id is not None:
            base_sql += " AND t.user_id = :user_id"
            params["user_id"] = user_id

        with self.engine.connect() as conn:
            df = pd.read_sql_query(text(base_sql), conn, params=params)
        return df
