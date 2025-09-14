from __future__ import annotations

from sqlalchemy import text

from category_lib import *
from database import *
from model.main import *
from utils import *


def classify_row(merchant: str, amount: int, kst_dt: datetime) -> CategoryHit:
    nm = merchant

    # NOTE: 1)
    for pat, sub_name in REGEX_RULES:
        if pat.search(nm):
            return CategoryHit(sub_name, "regex")

    hour = kst_dt.hour
    # NOTE: 2) 식비 세부 결정
    if CAFE_HINT.search(nm):
        return CategoryHit("커피", "heuristic:cafe")
        # return CategoryHit("음료(반복 구매)", "heuristic:drink")
    if CONVENIENCE_HINT.search(nm):
        return CategoryHit("간식", "heuristic:conv_store")
    if 11 <= hour <= 14 or 18 <= hour <= 21:
        if amount > 10000:
            return CategoryHit("외식", "heuristic:dining")

    # NOTE: 3) ??

    # NOTE: 4) ML 보조 판단
    init_ml_once()
    preds: List[Tuple[str, str]] = []

    if TFIDF_VEC is not None and TFIDF_CLF is not None:
        try:
            X = TFIDF_VEC.transform([nm])
            if hasattr(TFIDF_CLF, 'predict_proba'):
                p = TFIDF_CLF.predict_proba(X)[0]
                import numpy as np
                idx = int(np.argmax(p))
                conf = float(p[idx])
            else:
                # decision_function → softmax로 근사
                import numpy as np
                d = TFIDF_CLF.decision_function(X)[0]
                d = d - d.max()
                p = np.exp(d) / np.exp(d).sum()
                idx = int(np.argmax(p))
                conf = float(p[idx])
            # id→sub_name 매핑은 벡터화/학습 시 순서를 알 수 없으므로, 간단히 가장 가까운 sub_name을 추정
            # 여기서는 확률 top이 기존 규칙과 충돌하지 않는다고 가정하고, conf만 기준으로 사용
            # (정교한 label 매핑은 별도 train 스크립트에서 저장 권장)
            # 임시로 '기타' 반환 방지: conf만 체크하고 서브카테고리는 미지정 → 기타
            if conf >= ML_CONFIRM:
                preds.append(("기타" "ml:tfidf"))
        except Exception:
            pass
    if KOBERT_TOK is not None and KOBERT_MODEL is not None and KOBERT_CLF is not None:
        try:
            emb = kobert_embed([nm])
            p = KOBERT_CLF.predict_proba(emb)[0]
            import numpy as np
            idx = int(np.argmax(p))
            conf = float(p[idx])
            if conf >= ML_CONFIRM:
                preds.append(("기타", "ml:kobert"))
        except Exception:
            pass
    if preds:
        # 현재 버전은 ML로 서브카테고리 명까지 안정적으로 매핑하려면 학습 시 라벨 인덱스 ↔ sub_name 테이블이 필요함.
        # 해당 테이블을 아직 생성하지 않았으므로, ML은 신뢰도 판단용으로만 사용하고 최종 라벨은 보수적으로 기타 유지.
        best = sorted(preds, key=lambda x: x[1], reverse=True)[0]
        return CategoryHit(best[0], best[1])

    # NOTE: 5) 그래도 남은 것들 == 기타
    return CategoryHit("기타", "fallback")


# NOTE: 주 카테고라이징 로직
INSERT_SQL = text(
    """
    INSERT INTO transactions (
        user_id, sub_id, major_id, transacted_at,
        amount, merchanr_name, staus, created_at, updated_at
    ) VALUES (
        :user_id, :sub_id, :major_id, :transacted_at,
        :amount, :merchanr_name, :staus, :created_at, :updated_at
    )
    ON DUPLICATE KEY UPDATE
        amount = VALUES(amount),
        merchanr_name = VALUES(merchanr_name),
        staus = VALUES(staus),
        updated_at = VALUES(updated_at)
    """
)


def process_file(path: str, engine: Engine, sub_map: Dict[str, SubRow], unknown_log: List[dict]):
    m = re.search(r"(\d+)_거래내역\.xlsx$", os.path.basename(path))
    if not m:
        print(f"[SKIP] file name not matching pattern: {path}")
        return
    user_id = int(m.group(1))

    df = pd.read_excel(path, dtype=str)
    cols = {c.strip(): c for c in df.columns}
    need = ['거래일시', '적요', '출금액']
    for n in need:
        if n not in cols:
            raise KeyError(f"Column '{n}' not found in {path}. Found: {list(df.columns)}")

    now_kst = datetime.now(KST)

    rows = []
    for _, row in df.iterrows():
        raw_dt = row[cols['거래일시']]
        raw_nm = row[cols['적요']]
        raw_amt = row[cols['출금액']]

        try:
            kst_dt = parse_datetime_kst(raw_dt)
        except Exception:
            unknown_log.append({
                'user_id': user_id,
                'reason': 'bad_datetime',
                '거래일시': str(raw_dt),
                '적요': str(raw_nm),
                '출금액': str(raw_amt),
                'file': path,
            })
            continue
        merchant = normalize_merchant(raw_nm)
        amount = parse_amount(raw_amt)

        hit = classify_row(merchant, amount, kst_dt)
        sub_name = hit.sub_name

        # map sub_name -> (sub_id, major_id); fallback to 기타
        if sub_name not in sub_map:
            unknown_log.append({
                'user_id': user_id,
                'reason': 'unknown_sub_name',
                'sub_name': sub_name,
                'merchant': merchant,
                'amount': amount,
                'file': path,
            })
            sub_name = "기타" if "기타" in sub_map else list(sub_map.keys())[0]

        sub = sub_map[sub_name]
        status = '미반영'

        rows.append({
            'user_id': user_id,
            'sub_id': sub.sub_id,
            'major_id': sub.major_id,
            'transacted_at': kst_dt.astimezone(KST).replace(tzinfo=None),  # naive KST
            'amount': amount,
            'merchanr_name': merchant,
            'staus': status,
            'created_at': now_kst.replace(tzinfo=None),
            'updated_at': now_kst.replace(tzinfo=None),
        })

    if not rows:
        return

    with engine.begin() as conn:
        conn.execute(INSERT_SQL, rows)