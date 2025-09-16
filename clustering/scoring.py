from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_category_profile(tx: pd.DataFrame, assign: pd.DataFrame) -> pd.DataFrame:
    """
    tx: columns = [user_id, sub_id, transacted_at, amount, ...]
    assign: DataFrame({"user_id", "cluster"})  # KMeans 라벨 결과
    반환: cluster x sub_id 단위 프로파일(금액/건수/평균티켓/비중/최근14일 추세)
    """
    df = tx.copy()
    df["transacted_at"] = pd.to_datetime(df["transacted_at"])
    df = df.merge(assign, on="user_id", how="inner")

    grp = df.groupby(["cluster","sub_id"], as_index=False).agg(
        spend_30d=("amount","sum"),
        cnt=("amount","size"),
        avg_ticket=("amount","mean"),
    )
    totals = grp.groupby("cluster")["spend_30d"].sum().rename("cluster_total")
    grp = grp.merge(totals, on="cluster")
    grp["share_30d"] = grp["spend_30d"] / (grp["cluster_total"] + 1e-9)

    # 최근 14일 vs 직전 14일 추세
    tmax = df["transacted_at"].max()
    cut14, cut28 = tmax - pd.Timedelta(days=14), tmax - pd.Timedelta(days=28)
    last14 = df[df["transacted_at"] >= cut14].groupby(["cluster","sub_id"])["amount"].sum().rename("spend_last14")
    prev14 = df[(df["transacted_at"] < cut14) & (df["transacted_at"] >= cut28)] \
               .groupby(["cluster","sub_id"])["amount"].sum().rename("spend_prev14")
    grp = grp.merge(last14, on=["cluster","sub_id"], how="left") \
             .merge(prev14, on=["cluster","sub_id"], how="left") \
             .fillna({"spend_last14":0.0,"spend_prev14":0.0})
    grp["trend14_pct"] = np.where(
        grp["spend_prev14"] > 0,
        (grp["spend_last14"] - grp["spend_prev14"]) / grp["spend_prev14"],
        np.nan
    )
    return grp

def score_categories(profile: pd.DataFrame,
                     w_share=0.5, w_trend=0.3, w_ticket=0.2,
                     min_cnt=5, min_share=0.02) -> pd.DataFrame:
    """
    profile: cluster_category_profile() 결과
    가중합 점수로 절약 후보 카테고리를 상위 정렬해 반환
    """
    df = profile.copy()

    # 군집별 z-score 정규화
    def _z(g, col):
        s = g[col]
        return (s - s.mean()) / (s.std() + 1e-9)

    df["z_share"]  = df.groupby("cluster").apply(_z, "share_30d").reset_index(level=0, drop=True)
    df["z_trend"]  = df.groupby("cluster").apply(_z, "trend14_pct").reset_index(level=0, drop=True).fillna(0.0)
    df["z_ticket"] = df.groupby("cluster").apply(_z, "avg_ticket").reset_index(level=0, drop=True)

    df["score"] = w_share*df["z_share"] + w_trend*df["z_trend"] + w_ticket*df["z_ticket"]

    # 희소/미미한 카테고리 컷(노이즈 방지)
    df = df[(df["cnt"] >= min_cnt) & (df["share_30d"] >= min_share)]
    return df.sort_values(["cluster","score"], ascending=[True, False])
