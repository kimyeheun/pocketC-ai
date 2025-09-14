import numpy as np

from database import *


# 시간 파생변수
tx["transacted_at"] = pd.to_datetime(tx["transacted_at"])
tx["is_weekend"] = tx["transacted_at"].dt.weekday >= 5
tx["hour"] = tx["transacted_at"].dt.hour
tx["is_late_night"] = (tx["hour"] >= 22) | (tx["hour"] < 6)

# 사용자-소분류 단위 집계 (30일)
agg = tx.groupby(["user_id","sub_id"], as_index=False).agg(
    spend_30d=("amount","sum"),
    freq_30d=("amount","size"),
    avg_ticket=("amount","mean"),
    max_ticket=("amount","max"),
    late_night_share=("is_late_night","mean"),
    weekend_share=("is_weekend","mean")
)

# 전체 사용자-총지출
total_spend = tx.groupby("user_id")["amount"].sum().rename("total_spend_30d")
agg = agg.merge(total_spend, on="user_id")
agg["share_30d"] = agg["spend_30d"] / (agg["total_spend_30d"]+1e-9)

# Recency: 최근 7일 대비 30일
cut7 = tx["transacted_at"].max() - pd.Timedelta(days=7)
spend7 = tx[tx["transacted_at"] >= cut7].groupby(["user_id","sub_id"])["amount"].sum().rename("spend_7d")
agg = agg.merge(spend7, on=["user_id","sub_id"], how="left").fillna({"spend_7d":0})
agg["recency7d_share"] = agg["spend_7d"] / (agg["spend_30d"]+1e-9)

# Trend: 최근 14일 vs 이전 14일
cut14 = tx["transacted_at"].max() - pd.Timedelta(days=14)
cut28 = tx["transacted_at"].max() - pd.Timedelta(days=28)
spend_last14 = tx[tx["transacted_at"] >= cut14].groupby(["user_id","sub_id"])["amount"].sum().rename("spend_last14")
spend_prev14 = tx[(tx["transacted_at"] < cut14) & (tx["transacted_at"] >= cut28)] \
                 .groupby(["user_id","sub_id"])["amount"].sum().rename("spend_prev14")
agg = agg.merge(spend_last14, on=["user_id","sub_id"], how="left") \
         .merge(spend_prev14, on=["user_id","sub_id"], how="left") \
         .fillna(0)
agg["trend14d_pct"] = np.where(
    agg["spend_prev14"]>0,
    (agg["spend_last14"] - agg["spend_prev14"]) / agg["spend_prev14"],
    np.nan
)

# -------------------------------
# 3. 사용자 벡터 & 클러스터링
# -------------------------------
# (예: 사용자별 sub_id 피처를 pivot → wide form)
user_feat = agg.pivot_table(
    index="user_id",
    columns="sub_id",
    values="share_30d",
    fill_value=0
)

# 행태 피처(사용자 전체 기준 late_night, weekend 등) 추가
behavior = agg.groupby("user_id").agg(
    late_night_share_all=("late_night_share","mean"),
    weekend_share_all=("weekend_share","mean"),
    avg_ticket_all=("avg_ticket","mean")
)
user_feat = user_feat.merge(behavior, on="user_id")

print("user_feat : ", user_feat)
