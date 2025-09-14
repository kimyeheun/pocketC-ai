from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Transaction:
    user_id: int
    sub_id: int
    major_id: int
    transacted_at: datetime  # naive KST
    amount: int
    merchant_name: str
    status: str
    created_at: datetime
    updated_at: datetime
