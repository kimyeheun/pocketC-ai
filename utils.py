from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, date
import pytz

KST = pytz.timezone("Asia/Seoul")


def utc_to_kst(ts_utc: datetime) -> datetime:
    if ts_utc.tzinfo is None:
        ts_utc = ts_utc.replace(tzinfo=timezone.utc)
    return ts_utc.astimezone(KST)

def slug_to_name(slug: str) -> str:
    # TODO: categories 테이블에서 조회하도록 변경
    m = {
        "food.cafe_coffee": "커피",
        "food.convenience_store": "편의점",
        "food.delivery": "배달",
        "food.snack": "간식",
        "entertainment.game": "오락비"
    }
    return m.get(slug, slug.split(".")[-1])