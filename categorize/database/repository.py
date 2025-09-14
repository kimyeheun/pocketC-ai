from __future__ import annotations

from typing import Dict, List, Protocol

from categorize.category_lib import SubCategory
from categorize.domain.transaction import Transaction


class SubCategoryRepository(Protocol):
    def load_map(self) -> Dict[str, SubCategory]: ...

class TransactionRepository(Protocol):
    def insert_many(self, rows: List[Transaction]) -> None: ...
