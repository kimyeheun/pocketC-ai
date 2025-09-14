from __future__ import annotations

from typing import Optional, List

from categorize.category_lib import CategoryHit
from categorize.model.main import predict_candidates, get_label_names
from categorize.rules.base import Rule


class MLRule(Rule):
    def __init__(self, label_inverse_map: Optional[List[str]] = None):
        self.label_inverse_map = label_inverse_map or (get_label_names() or None)

    def apply(self, merchant: str, amount: int, kst_hour: int) -> Optional[CategoryHit]:
        cands = predict_candidates(merchant, self.label_inverse_map)
        if cands:
            sub_name, source, _ = max(cands, key=lambda x: x[2])
            return CategoryHit(sub_name, source)
        return None
