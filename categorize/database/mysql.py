from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from categorize.category_lib import SubCategory
from categorize.database.repository import SubCategoryRepository, TransactionRepository
from categorize.domain.transaction import Transaction


class MySQLSubCategoryRepository(SubCategoryRepository):
    def __init__(self, engine: Engine):
        self.engine = engine

    def load_map(self) -> Dict[str, SubCategory]:
        df = pd.read_sql("SELECT sub_id, major_id, sub_name FROM sub_categories", self.engine)
        mapping: Dict[str, SubCategory] = {}
        for _, r in df.iterrows():
            name = str(r["sub_name"]).strip()
            if name not in mapping:
                mapping[name] = SubCategory(int(r["sub_id"]), int(r["major_id"]), name)
        return mapping

INSERT_SQL = text("""
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
""")

class MySQLTransactionRepository(TransactionRepository):
    def __init__(self, engine: Engine):
        self.engine = engine

    def insert_many(self, rows: List[Transaction]) -> None:
        if not rows: return
        payload = [dict(
            user_id=t.user_id, sub_id=t.sub_id, major_id=t.major_id,
            transacted_at=t.transacted_at, amount=t.amount,
            merchanr_name=t.merchant_name, staus=t.status,
            created_at=t.created_at, updated_at=t.updated_at
        ) for t in rows]
        with self.engine.begin() as conn:
            conn.execute(INSERT_SQL, payload)
