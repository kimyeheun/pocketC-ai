from __future__ import annotations
import re
from typing import List, Tuple, Optional
from categorize.rules.base import Rule
from categorize.category_lib import CategoryHit

class RegexRule(Rule):
    def __init__(self, patterns: List[Tuple[re.Pattern, str]]):
        self.patterns = patterns

    def apply(self, merchant: str, amount: int, kst_hour: int) -> Optional[CategoryHit]:
        for pat, sub_name in self.patterns:
            if pat.search(merchant):
                return CategoryHit(sub_name, "regex")
        return None
