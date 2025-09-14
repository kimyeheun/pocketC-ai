from __future__ import annotations
from typing import List
from categorize.category_lib import CategoryHit
from categorize.rules.base import Rule

class Categorizer:
    def __init__(self, rules: List[Rule], fallback: str = "기타"):
        self.rules = rules
        self.fallback = fallback

    def classify(self, merchant: str, amount: int, kst_hour: int) -> CategoryHit:
        for r in self.rules:
            hit = r.apply(merchant, amount, kst_hour)
            if hit:
                return hit
        return CategoryHit(self.fallback, "fallback")
