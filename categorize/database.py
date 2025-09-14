from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@dataclass
class SubRow:
    sub_id: int
    major_id: int
    sub_name: str


def make_engine_from_env() -> Engine:
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "0000")
    name = os.getenv("DB_NAME", "pocketc")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True, future=True)


def load_sub_map(engine: Engine) -> Dict[str, SubRow]:
    sql = "SELECT sub_id, major_id, sub_name FROM sub_categories"
    df = pd.read_sql(sql, engine)
    mapping: Dict[str, SubRow] = {}
    for _, r in df.iterrows():
        name = str(r["sub_name"]).strip()
        if name in mapping:
            print(f"[WARN] Duplicate sub_name in DB: {name} (keeping first)")
            continue
        mapping[name] = SubRow(int(r["sub_id"]), int(r["major_id"]), name)
    return mapping
