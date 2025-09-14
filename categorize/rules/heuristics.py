from __future__ import annotations
from typing import Optional
from categorize.rules.base import Rule
from categorize.category_lib import CategoryHit
from categorize.category_lib import CAFE_HINT, CONVENIENCE_HINT  # 패턴 재사용

class CafeRule(Rule):
    def apply(self, merchant: str, amount: int, kst_hour: int) -> Optional[CategoryHit]:
        if CAFE_HINT.search(merchant):
            return CategoryHit("커피", "heuristic:cafe")
        return None

class ConvenienceRule(Rule):
    def apply(self, merchant: str, amount: int, kst_hour: int) -> Optional[CategoryHit]:
        if CONVENIENCE_HINT.search(merchant):
            return CategoryHit("간식", "heuristic:conv_store")
        return None

class DiningTimeRule(Rule):
    def apply(self, merchant: str, amount: int, kst_hour: int) -> Optional[CategoryHit]:
        if (11 <= kst_hour <= 14 or 18 <= kst_hour <= 21) and amount > 10_000:
            return CategoryHit("외식", "heuristic:dining")
        return None
