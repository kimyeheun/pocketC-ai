from __future__ import annotations

from datetime import datetime

from categorize.category_lib import CategoryHit
from categorize.category_lib import REGEX_RULES
from categorize.rules.chain import Categorizer
from categorize.rules.heuristics import CafeRule, ConvenienceRule, DiningTimeRule
from categorize.rules.ml_rule import MLRule
from categorize.rules.regex_rule import RegexRule


class CategorizationService:
    def __init__(self, fallback: str = "기타"):
        self._categorizer = Categorizer(
            rules=[
                RegexRule(REGEX_RULES),
                CafeRule(),
                ConvenienceRule(),
                DiningTimeRule(),
                MLRule(),
            ],
            fallback=fallback,
        )

    def classify(self, merchant: str, amount: int, kst_dt: datetime) -> CategoryHit:
        return self._categorizer.classify(merchant, amount, kst_dt.hour)
