from __future__ import annotations

import os
import pandas as pd
import sys
import time

from common.engine import make_engine_from_env
from categorize.database.mysql import MySQLSubCategoryRepository, MySQLTransactionRepository
from categorize.service.transaction_ingestion import IngestionService


def main():
    if len(sys.argv) < 2:
        print("환경변수 또는 인자로 폴더 경로를 전달하세요.")
        sys.exit(1)

    folder = sys.argv[1]
    engine = make_engine_from_env()

    sub_repo = MySQLSubCategoryRepository(engine)
    sub_map = sub_repo.load_map()
    if "기타" not in sub_map:
        print("[WARN] sub_categories does not contain '기타'. Fallback will pick an arbitrary sub_id.")

    tx_repo = MySQLTransactionRepository(engine)
    svc = IngestionService(sub_map, tx_repo, fallback_name="기타")

    unknown_log = svc.process_folder(folder)

    if unknown_log:
        out = os.path.join(folder, f"_curation_queue_{int(time.time())}.csv")
        pd.DataFrame(unknown_log).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[INFO] Curation queue written: {out}")
    else:
        print("[INFO] 다 잘 카테고라이징 되었습니다~.")

    # NOTE: user 별 카테고리 사용량 시각화


if __name__ == "__main__":
    main()
