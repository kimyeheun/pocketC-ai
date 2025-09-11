from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
from dateutil import tz

KST = tz.gettz("Asia/Seoul")


# NOTE: 출금액, 시간 범용성 높이기
def parse_amount(x) -> int:
    if pd.isna(x):
        return 0
    s = str(x).strip()
    s = s.replace("-", "").replace(",", "").replace("원", "")
    try:
        return int(float(s))
    except Exception:
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else 0


# NOTE: 결제 일시 파싱
def parse_datetime_kst(x) -> datetime:
    if isinstance(x, datetime):
        dt = x
    else:
        s = str(x).strip()
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y.%m.%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y.%m.%d %H:%M",
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ]:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except Exception:
                dt = None
        if dt is None:
            raise ValueError(f"Unrecognized datetime format: {x}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt

# NOTE: 적요 정규화
BRACKET_RE = re.compile(r"[\[\(\{].*?[\]\)\}]|")
NON_ALNUM_KO_EN = re.compile(r"[^0-9A-Za-z가-힣\s]", re.UNICODE)
BRANCH_SUFFIX_RE = re.compile(r"(역|점|센터|점포|지점|영업점|사옥|타워)\s*\w*$")


def normalize_merchant(raw: str) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw)
    s = s.replace("\u200b", "")  # zero-width
    s = BRACKET_RE.sub(" ", s)
    s = NON_ALNUM_KO_EN.sub(" ", s)
    s = BRANCH_SUFFIX_RE.sub("", s)
    s = s.replace(" ","")
    return s

