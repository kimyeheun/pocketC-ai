from __future__ import annotations
import os, re, pandas as pd
from datetime import datetime
from typing import Dict, List
from categorize.utils import parse_amount, parse_datetime_kst, normalize_merchant, KST
from categorize.category_lib import SubCategory
from categorize.domain.transaction import Transaction
from categorize.service.categorization import CategorizationService

USER_FILE_RE = re.compile(r"(\d+)_거래내역\.xlsx$")

class IngestionService:
    def __init__(self, sub_map: Dict[str, SubCategory], tx_repo, fallback_name: str = "기타"):
        self.sub_map = sub_map
        self.tx_repo = tx_repo
        self.categorizer = CategorizationService(fallback=fallback_name)

    def process_folder(self, folder: str) -> List[dict]:
        unknown_log: List[dict] = []
        excel_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_거래내역.xlsx")]
        excel_paths.sort()
        for p in excel_paths:
            self._process_file(p, unknown_log)
        return unknown_log

    def _process_file(self, path: str, unknown_log: List[dict]):
        m = USER_FILE_RE.search(os.path.basename(path))
        print(m)
        if not m:
            print(f"[SKIP] file name not matching pattern: {path}")
            return
        user_id = int(m.group(1))
        df = pd.read_excel(path, dtype=str)

        cols = {c.strip(): c for c in df.columns}
        need = ["거래일시", "적요", "출금액"]
        for n in need:
            if n not in cols:
                raise KeyError(f"Column '{n}' not found in {path}. Found: {list(df.columns)}")

        now_kst = datetime.now(KST)
        rows: List[Transaction] = []

        for _, row in df.iterrows():
            raw_dt, raw_nm, raw_amt = row[cols["거래일시"]], row[cols["적요"]], row[cols["출금액"]]
            try:
                kst_dt = parse_datetime_kst(raw_dt)
            except Exception:
                unknown_log.append({"user_id": user_id, "reason": "bad_datetime",
                                    "거래일시": str(raw_dt), "적요": str(raw_nm), "출금액": str(raw_amt), "file": path})
                continue

            merchant = normalize_merchant(raw_nm)
            amount = parse_amount(raw_amt)
            hit = self.categorizer.classify(merchant, amount, kst_dt)
            sub_name = hit.sub_name

            if sub_name not in self.sub_map:
                unknown_log.append({"user_id": user_id, "reason": "unknown_sub_name",
                                    "sub_name": sub_name, "merchant": merchant, "amount": amount, "file": path})
                sub_name = "기타" if "기타" in self.sub_map else next(iter(self.sub_map.keys()))

            sub = self.sub_map[sub_name]
            rows.append(Transaction(
                user_id=user_id,
                sub_id=sub.sub_id,
                major_id=sub.major_id,
                transacted_at=kst_dt.astimezone(KST).replace(tzinfo=None),
                amount=amount,
                merchant_name=merchant,
                status="미반영",
                created_at=now_kst.replace(tzinfo=None),
                updated_at=now_kst.replace(tzinfo=None),
            ))
        self.tx_repo.insert_many(rows)
