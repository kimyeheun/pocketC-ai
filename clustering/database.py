import os

import pandas as pd
from sqlalchemy import create_engine


def make_engine_from_env():
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "0000")
    name = os.getenv("DB_NAME", "pocketc")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True, future=True)

engine = make_engine_from_env()

# 1. 최근 30일치 거래내역 불러오기
# TODO: 해당 달 거래내역으로 불러오기
query = """
SELECT 
    transaction_id, user_id, s.sub_name as sub_id, m.major_name as major_id, 
    transacted_at, amount, merchanr_name, staus
FROM transactions as t
JOIN sub_categories as s on t.sub_id = s.sub_id
JOIN major_categories as m on m.major_id = s.major_id
WHERE transacted_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
"""
tx = pd.read_sql(query, engine)
