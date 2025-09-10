from fastapi import FastAPI, HTTPException
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, date
import pymysql
import pytz
import json
import statistics
from Model import *

app = FastAPI(title="pocketC AI")


# ========= 통계/후보 계산 =========
def choose_need_category(conn, user_id: int) -> Optional[str]:
    """
    최근 7일 vs 직전 23일 증가율 + 최근 비중으로 '필요 카테고리' 추정 (간단 휴리스틱)
    """
    with conn.cursor() as cur:
        q = """
        WITH base AS (
          SELECT DATE(CONVERT_TZ(transacted_at,'UTC','Asia/Seoul')) AS d,
                 sub_category AS slug, SUM(amount) AS amt
          FROM transactions
          WHERE user_id=%s AND status='반영'
            AND transacted_at >= UTC_TIMESTAMP() - INTERVAL 30 DAY
          GROUP BY d, slug
        ),
        win AS (
          SELECT slug,
                 SUM(CASE WHEN d >= (CURRENT_DATE() - INTERVAL 7 DAY) THEN amt ELSE 0 END) AS sum7,
                 SUM(CASE WHEN d <  (CURRENT_DATE() - INTERVAL 7 DAY) THEN amt ELSE 0 END) AS sum23
          FROM base GROUP BY slug
        )
        SELECT slug
        FROM win
        ORDER BY (sum7 / NULLIF(sum23/23*7, 0)) DESC
        LIMIT 1;
        """
        cur.execute(q, (user_id,))
        row = cur.fetchone()
        return row["slug"] if row else None

def fetch_stats_python(conn, user_id: int, category_slug: str) -> Dict[str, Any]:
    """
    최근 30일 사용자-카테고리 데이터를 가져와 파이썬에서 분위/평균/최빈시간을 계산.
    (MySQL의 퍼센타일 함수 호환성을 피해 안전하게 처리)
    """
    with conn.cursor() as cur:
        q = """
        SELECT
          amount,
          HOUR(CONVERT_TZ(transacted_at,'UTC','Asia/Seoul')) AS hr,
          DATE(CONVERT_TZ(transacted_at,'UTC','Asia/Seoul')) AS d
        FROM transactions
        WHERE user_id=%s AND sub_category=%s AND status='반영'
          AND transacted_at >= UTC_TIMESTAMP() - INTERVAL 30 DAY
        """
        cur.execute(q, (user_id, category_slug))
        rows = cur.fetchall()

    amounts = [r["amount"] for r in rows]
    by_day: Dict[date, List[int]] = {}
    hours: Dict[int, int] = {}
    for r in rows:
        by_day.setdefault(r["d"], []).append(r["amount"])
        hours[r["hr"]] = hours.get(r["hr"], 0) + 1

    def percentile(data: List[int], p: float) -> int:
        if not data:
            return 0
        data = sorted(data)
        k = (len(data)-1) * p
        f = int(k)
        c = min(f+1, len(data)-1)
        if f == c: return int(data[int(k)])
        return int(round(data[f] + (data[c]-data[f])*(k-f)))

    per_txn_p40 = percentile(amounts, 0.40) if amounts else 0
    mean_daily_count = statistics.mean([len(v) for v in by_day.values()]) if by_day else 0
    daily_sum = [sum(v) for v in by_day.values()]
    daily_sum_p75 = percentile(daily_sum, 0.75) if daily_sum else 0

    # 최빈 시간대 → 3시간 블록으로 매끄럽게
    peak_hour = max(hours.items(), key=lambda x: x[1])[0] if hours else None
    peak_window = None
    if peak_hour is not None:
        start = f"{peak_hour:02d}:00"
        end_h = (peak_hour + 3) % 24
        end = f"{end_h:02d}:00"
        peak_window = {"start": start, "end": end}

    return {
        "category_slug": category_slug,
        "per_txn_p40": per_txn_p40,
        "mean_daily_count": mean_daily_count,
        "daily_sum_p75": daily_sum_p75,
        "peak_window": peak_window
    }

def param_candidates_from_stats(stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if stats["per_txn_p40"]:
        cand = int((stats["per_txn_p40"] // 100) * 100)
        low, high = max(2000, cand-1000), cand+1000
        out["PER_TXN_CAP_DAILY"] = {"per_txn": {"cand": cand, "range": [low, high]}}
    if stats["mean_daily_count"]:
        m = stats["mean_daily_count"]
        cand = max(0, int(m) - 1)
        out["COUNT_CAP_DAILY"] = {"N": {"cand": cand, "range": [max(0,cand-1), cand+1]}}
    if stats["daily_sum_p75"]:
        cand = int((stats["daily_sum_p75"] // 100) * 100)
        out["SPEND_CAP_DAILY"] = {"amount": {"cand": cand, "range": [max(0,cand-3000), cand+3000]}}
    if stats["peak_window"]:
        out["TIME_WINDOW_BAN_DAILY"] = {"time_window": {"cand": {**stats["peak_window"], "tz":"Asia/Seoul"}}}
    out["BAN_CATEGORY_DAILY"] = {}
    return out

def pick_template(preferences: List[str], param_cands: Dict[str, Dict[str, Any]]) -> str:
    gentle = "gentle_first" in preferences
    order = ["PER_TXN_CAP_DAILY","COUNT_CAP_DAILY","SPEND_CAP_DAILY","TIME_WINDOW_BAN_DAILY","BAN_CATEGORY_DAILY"] if gentle \
            else ["SPEND_CAP_DAILY","COUNT_CAP_DAILY","PER_TXN_CAP_DAILY","TIME_WINDOW_BAN_DAILY","BAN_CATEGORY_DAILY"]
    for t in order:
        if t in param_cands:
            return t
    return "PER_TXN_CAP_DAILY"

# ========= DSL/플랜 빌더 =========
def build_dsl_and_plan(template_name: str, category_slug: str, params: Dict[str, Any]):
    if template_name == "BAN_CATEGORY_DAILY":
        dsl = {"all":[{"category_is": category_slug}]}
        plan = {"kind":"event_violation_only","success_mode":"eod_no_violation",
                "ops":[{"when":{"all":[{"eq":{"tx.category_slug":category_slug}}]},
                        "decision":"fail",
                        "explain":{"reason":"ban_category","label":category_slug}}]}
    elif template_name == "PER_TXN_CAP_DAILY":
        cap = int(params["per_txn"])
        dsl = {"all":[{"category_is":category_slug},{"amount_gt":cap}]}
        plan = {"kind":"event_violation_only","success_mode":"eod_no_violation",
                "ops":[{"when":{"all":[{"eq":{"tx.category_slug":category_slug}},
                                       {"gt":{"tx.amount":cap}}]},
                        "decision":"fail",
                        "explain":{"reason":"per_txn_cap","cap":cap}}]}
    elif template_name == "COUNT_CAP_DAILY":
        N = int(params["N"])
        dsl = {"all":[{"category_is":category_slug},{"daily_count_after_txn_gt":N}]}
        plan = {"kind":"event_violation_only","success_mode":"eod_no_violation","needs_counters":True,
                "ops":[{"when":{"all":[{"eq":{"tx.category_slug":category_slug}},
                                       {"gt":{"counters.daily_count_after_txn":N}}]},
                        "decision":"fail",
                        "explain":{"reason":"daily_count_cap","cap":N}}]}
    elif template_name == "SPEND_CAP_DAILY":
        amt = int(params["amount"])
        dsl = {"all":[{"category_is":category_slug},{"daily_sum_after_txn_gt":amt}]}
        plan = {"kind":"event_violation_only","success_mode":"eod_no_violation","needs_counters":True,
                "ops":[{"when":{"all":[{"eq":{"tx.category_slug":category_slug}},
                                       {"gt":{"counters.daily_sum_after_txn":amt}}]},
                        "decision":"fail",
                        "explain":{"reason":"daily_sum_cap","cap":amt}}]}
    else:  # TIME_WINDOW_BAN_DAILY
        tw = params["time_window"]
        dsl = {"all":[{"category_is":category_slug},{"time_in":tw}]}
        plan = {"kind":"event_violation_only","success_mode":"eod_no_violation",
                "ops":[{"when":{"all":[{"eq":{"tx.category_slug":category_slug}},
                                       {"time_in":tw}]},
                        "decision":"fail",
                        "explain":{"reason":"time_ban","window":f"{tw['start']}-{tw['end']}"}}]}
    return dsl, plan

def render_text(template_name: str, slug: str, params: Dict[str, Any]) -> str:
    label = slug_to_name(slug)
    if template_name == "BAN_CATEGORY_DAILY":
        return f"오늘 {label} 안 사기"
    if template_name == "PER_TXN_CAP_DAILY":
        return f"{label} 1건 {int(params['per_txn']):,}원 이하"
    if template_name == "COUNT_CAP_DAILY":
        return f"오늘 {label} {int(params['N'])}회 이하"
    if template_name == "SPEND_CAP_DAILY":
        return f"오늘 {label} {int(params['amount']):,}원 이하"
    if template_name == "TIME_WINDOW_BAN_DAILY":
        tw = params["time_window"]; tl = f"{tw['start']}–{tw['end']}"
        return f"{tl} {label} 금지"
    return ""

# ========= DB 저장/조회 =========
def insert_mission_instance(conn, user_id: int, template_name: str, category_slug: str,
                            params: Dict[str, Any], dsl: Dict[str, Any],
                            compiled_plan: Dict[str, Any], render_str: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT template_id, template_version FROM mission_templates WHERE name=%s ORDER BY template_id DESC LIMIT 1",
                    (template_name,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=400, detail=f"템플릿 {template_name} 없음")
        tid, tver = row["template_id"], row["template_version"]
        cur.execute("""
            INSERT INTO mission_instances
            (user_id, template_id, template_version, category_slug, params, dsl_instance, compiled_plan, render_str, status, valid_from)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'active',CURRENT_DATE())
        """, (user_id, tid, tver, category_slug, json.dumps(params,ensure_ascii=False),
              json.dumps(dsl,ensure_ascii=False), json.dumps(compiled_plan,ensure_ascii=False), render_str))
        conn.commit()
        return cur.lastrowid

def load_active_missions(conn, user_id: int, date_kst: date) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("""
        SELECT mission_id, category_slug, compiled_plan
        FROM mission_instances
        WHERE user_id=%s AND status='active'
          AND %s BETWEEN valid_from AND COALESCE(valid_to, %s)
        """, (user_id, date_kst, date_kst))
        rows = cur.fetchall()
        for r in rows:
            r["compiled_plan"] = json.loads(r["compiled_plan"])
        return rows

def read_daily_counters(conn, user_id: int, date_kst: date, category_slug: str) -> Tuple[int,int]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT txn_count, amount_sum
            FROM daily_user_category_stats
            WHERE date_kst=%s AND user_id=%s AND category_slug=%s
            FOR UPDATE
        """, (date_kst, user_id, category_slug))
        row = cur.fetchone()
        return (row["txn_count"], row["amount_sum"]) if row else (0,0)

def update_daily_counters(conn, user_id: int, date_kst: date, category_slug: str, amount: int):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO daily_user_category_stats(date_kst, user_id, category_slug, txn_count, amount_sum)
        VALUES (%s,%s,%s,1,%s)
        ON DUPLICATE KEY UPDATE
          txn_count = txn_count + 1,
          amount_sum = amount_sum + VALUES(amount_sum)
        """, (date_kst, user_id, category_slug, amount))

def write_evaluation_fail(conn, mission_id: int, tx_id: int, ts_utc: datetime, explain: Dict[str,Any]):
    with conn.cursor() as cur:
        cur.execute("""
        INSERT IGNORE INTO mission_evaluations(mission_id, txn_id, event_ts, decision, explain_json)
        VALUES (%s, %s, %s, 'fail', %s)
        """, (mission_id, str(tx_id), ts_utc, json.dumps(explain,ensure_ascii=False)))

# ========= 플랜 실행기 =========
def time_in_window(local_dt: datetime, tw: Dict[str,str]) -> bool:
    t = local_dt.time()
    s_h, s_m = map(int, tw["start"].split(":"))
    e_h, e_m = map(int, tw["end"].split(":"))
    start = t.replace(hour=s_h, minute=s_m, second=0, microsecond=0)
    end   = t.replace(hour=e_h, minute=e_m, second=0, microsecond=0)
    if tw.get("tz") and tw["tz"] != "Asia/Seoul":
        # 본 예시는 KST로만 평가하므로 tz는 표기만 사용
        pass
    if s_h < e_h or (s_h==e_h and s_m<e_m):
        return start <= t < end
    else:
        # 오버나이트 (예: 22:00~06:00)
        return (t >= start) or (t < end)

def execute_plan_for_mission(mission: Dict[str,Any],
                             tx: TransactionIn,
                             local_dt: datetime,
                             prev_count: int,
                             prev_sum: int) -> Tuple[str, Dict[str,Any]]:
    plan = mission["compiled_plan"]
    # 컨텍스트
    tx_ctx = {"category_slug": tx.sub_category, "amount": tx.amount}
    counters_after = {
        "daily_count_after_txn": prev_count + (1 if tx.sub_category == mission["category_slug"] else 0),
        "daily_sum_after_txn": prev_sum + (tx.amount if tx.sub_category == mission["category_slug"] else 0),
    }

    for op in plan.get("ops", []):
        cond = op["when"]
        all_ok = True
        for clause in cond.get("all", []):
            # eq
            if "eq" in clause:
                (left, right), = clause["eq"].items()
                if left == "tx.category_slug":
                    all_ok &= (tx_ctx["category_slug"] == right)
                else:
                    all_ok &= False
            # gt
            elif "gt" in clause:
                (left, thr), = clause["gt"].items()
                if left == "tx.amount":
                    all_ok &= (tx_ctx["amount"] > thr)
                elif left == "counters.daily_count_after_txn":
                    all_ok &= (counters_after["daily_count_after_txn"] > thr)
                elif left == "counters.daily_sum_after_txn":
                    all_ok &= (counters_after["daily_sum_after_txn"] > thr)
                else:
                    all_ok &= False
            # time_in
            elif "time_in" in clause:
                all_ok &= time_in_window(local_dt, clause["time_in"])
            else:
                all_ok = False
            if not all_ok: break

        if all_ok:
            # 현재는 FAIL 조건만 정의
            if op["decision"] == "fail":
                return "fail", op.get("explain", {})
    return "irrelevant", {}

# ========= 엔드포인트 =========

@app.post("/ai/missions/auto-generate", response_model=MissionInstanceOut)
def auto_generate(req: GenerateReq):
    conn = get_conn()
    try:
        with conn:
            slug = req.category_slug or choose_need_category(conn, req.user_id)
            if not slug:
                raise HTTPException(status_code=404, detail="생성할 카테고리를 찾지 못함")

            stats = fetch_stats_python(conn, req.user_id, slug)
            param_cands = param_candidates_from_stats(stats)
            template = pick_template(["gentle_first"], param_cands)

            params: Dict[str, Any] = {}
            if template in param_cands and param_cands[template]:
                for k, v in param_cands[template].items():
                    params[k] = v["cand"] if isinstance(v, dict) and "cand" in v else v

            dsl, plan = build_dsl_and_plan(template, slug, params)
            render = render_text(template, slug, params)
            mission_id = insert_mission_instance(conn, req.user_id, template, slug, params, dsl, plan, render)

            return MissionInstanceOut(
                mission_id=mission_id, user_id=req.user_id,
                template_name=template, category_slug=slug,
                params=params, render_str=render, dsl_instance=dsl
            )
    finally:
        conn.close()

@app.post("/ai/missions/evaluate-transaction")
def evaluate_transaction(tx: TransactionIn):
    if tx.status != "반영":
        return {"processed": False, "reason": "미반영"}

    local_dt = utc_to_kst(tx.transacted_at)
    date_kst = local_dt.date()
    conn = get_conn()

    try:
        with conn:
            missions = load_active_missions(conn, tx.user_id, date_kst)
            results = []
            # 카테고리별로 카운터 잠금/업데이트 (해당 tx.sub_category만)
            prev_count, prev_sum = read_daily_counters(conn, tx.user_id, date_kst, tx.sub_category or "")
            for m in missions:
                decision, explain = execute_plan_for_mission(m, tx, local_dt, prev_count, prev_sum)
                if decision == "fail":
                    write_evaluation_fail(conn, m["mission_id"], tx.transaction_id, tx.transacted_at, explain)
                results.append({"mission_id": m["mission_id"], "decision": decision, "explain": explain})

            # 러닝 카운터 업데이트 (tx 카테고리만 가산)
            if tx.sub_category:
                update_daily_counters(conn, tx.user_id, date_kst, tx.sub_category, tx.amount)

            conn.commit()
            return {"processed": True, "results": results}
    finally:
        conn.close()
