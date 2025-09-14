from __future__ import annotations

import os
import sys
import time
from typing import List

import pandas as pd

from categorization import process_file
from database import make_engine_from_env, load_sub_map


def main():
    if len(sys.argv) < 2:
        print("환경변수로 폴더 위치를 넣어야 합니다.")
        sys.exit(1)

    folder = sys.argv[1]
    engine = make_engine_from_env()
    sub_map = load_sub_map(engine)
    if "기타" not in sub_map:
        print("[WARN] sub_categories does not contain '기타'. Fallback will pick an arbitrary sub_id.")

    # Collect files
    excel_paths = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith("_거래내역.xlsx")
    ]
    excel_paths.sort()

    unknown_log: List[dict] = []
    for p in excel_paths:
        print(f"[INFO] 진행 중 : {p}")
        try:
            process_file(p, engine, sub_map, unknown_log)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            unknown_log.append({"file": p, "reason": "exception", "detail": str(e)})

    # Save curation queue
    if unknown_log:
        out = os.path.join(folder, f"_curation_queue_{int(time.time())}.csv")
        pd.DataFrame(unknown_log).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[INFO] Curation queue written: {out}")
    else:
        print("[INFO] 다 잘 카테고라이징 되었습니다~.")


if __name__ == "__main__":
    main()
