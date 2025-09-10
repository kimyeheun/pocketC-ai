from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, date

class GenerateReq(BaseModel):
    user_id: int
    category_slug: Optional[str] = None  # 없으면 자동 추천(최근 과다 카테고리)

class MissionInstanceOut(BaseModel):
    mission_id: int
    user_id: int
    template_name: str
    category_slug: str
    params: Dict[str, Any]
    render_str: str
    dsl_instance: Dict[str, Any]

class TransactionIn(BaseModel):
    transaction_id: int
    user_id: int
    transacted_at: datetime   # UTC로 가정
    amount: int
    merchant_name: Optional[str] = None
    major_category: Optional[str] = None
    sub_category: Optional[str] = None  # 슬러그라고 가정
    status: str  # "반영" | "미반영"