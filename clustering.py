import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==== 0) Synthetic sample of a single user's 30-day transactions ====
rng = np.random.default_rng(42)
now = datetime(2025, 9, 10, 10, 0, 0)

user_id = "user_123"

# Define subcategories we'll actually use for the demo (✅ ones)
subcats = [
    ("식비","커피"), ("식비","음료(반복 구매)"), ("식비","술"),
    ("식비","간식"), ("식비","배달음식"), ("식비","외식"), ("식비","식재료"),
    ("생활","생활용품"),
    ("통신/인터넷","OTT/구독서비스"),
    ("건강/의료","헬스/PT"),
    ("교육/자기계발","도서/교재"),
    ("쇼핑","의류/패션"), ("쇼핑","뷰티/미용"), ("쇼핑","온라인 쇼핑몰"), ("쇼핑","충동구매"),
    ("문화/여가","영화/공연"), ("문화/여가","게임/콘텐츠"), ("문화/여가","취미/오락"),
    ("문화/여가","여행"),
    ("기타","기타"),
]

merchants_by_sub = {
    ("식비","커피"): ["스타벅스", "이디야", "투썸"],
    ("식비","음료(반복 구매)"): ["공차", "탐앤탐스"],
    ("식비","술"): ["편의점 주류", "이자카야"],
    ("식비","간식"): ["파리바게뜨", "던킨"],
    ("식비","배달음식"): ["배달의민족", "요기요"],
    ("식비","외식"): ["김밥천국", "역전우동", "브런치카페"],
    ("식비","식재료"): ["이마트", "홈플러스"],
    ("생활","생활용품"): ["다이소", "쿠팡 로켓"],
    ("통신/인터넷","OTT/구독서비스"): ["넷플릭스", "웨이브", "티빙"],
    ("건강/의료","헬스/PT"): ["헬스장", "PT샵"],
    ("교육/자기계발","도서/교재"): ["교보문고", "알라딘"],
    ("쇼핑","의류/패션"): ["무신사", "지그재그"],
    ("쇼핑","뷰티/미용"): ["올리브영", "명동화장품"],
    ("쇼핑","온라인 쇼핑몰"): ["쿠팡", "11번가", "G마켓"],
    ("쇼핑","충동구매"): ["랜덤샵"],
    ("문화/여가","영화/공연"): ["CGV", "메가박스"],
    ("문화/여가","게임/콘텐츠"): ["스팀", "구글플레이"],
    ("문화/여가","취미/오락"): ["프라모델샵", "보드게임카페"],
    ("문화/여가","여행"): ["에어비앤비", "KTX"],
    ("기타","기타"): ["기타지출"],
}

def sample_txn_times(n, late_night_ratio=0.25, weekend_ratio=0.45):
    times = []
    for _ in range(n):
        # choose day within last 30 days
        day_offset = rng.integers(0, 30)
        d = now - timedelta(days=int(day_offset))
        # weekend?
        if rng.random() < weekend_ratio:
            # push to nearest weekend day (Sat/Sun)
            wd = d.weekday()  # Mon=0
            if wd < 5:
                d = d + timedelta(days=(5 - wd))  # to Saturday
        # time of day
        if rng.random() < late_night_ratio:
            hour = rng.integers(22, 24) if rng.random() < 0.7 else rng.integers(0, 6)
        else:
            # daytime spread
            hour = rng.integers(8, 22)
        minute = rng.integers(0, 60)
        times.append(d.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0))
    return times

def sample_amount(subcat):
    # rough ranges per subcategory
    ranges = {
        "커피": (4500, 6500),
        "음료(반복 구매)": (4000, 7000),
        "술": (15000, 40000),
        "간식": (3000, 7000),
        "배달음식": (15000, 30000),
        "외식": (10000, 35000),
        "식재료": (12000, 45000),
        "생활용품": (7000, 30000),
        "OTT/구독서비스": (9900, 19900),
        "헬스/PT": (40000, 120000),
        "도서/교재": (10000, 35000),
        "의류/패션": (30000, 150000),
        "뷰티/미용": (15000, 80000),
        "온라인 쇼핑몰": (10000, 100000),
        "충동구매": (5000, 50000),
        "영화/공연": (12000, 35000),
        "게임/콘텐츠": (10000, 40000),
        "취미/오락": (10000, 60000),
        "여행": (50000, 400000),
        "기타": (5000, 80000),
    }
    low, high = ranges.get(subcat, (10000, 30000))
    return int(rng.uniform(low, high))

# Build a biased distribution to reflect a "커피+배달 헤비" user
weights = {
    ("식비","커피"): 18, ("식비","배달음식"): 12, ("식비","외식"): 8, ("식비","간식"): 8,
    ("식비","음료(반복 구매)"): 5, ("식비","술"): 3, ("식비","식재료"): 6,
    ("생활","생활용품"): 4,
    ("통신/인터넷","OTT/구독서비스"): 3,
    ("건강/의료","헬스/PT"): 2,
    ("교육/자기계발","도서/교재"): 2,
    ("쇼핑","의류/패션"): 2, ("쇼핑","뷰티/미용"): 2, ("쇼핑","온라인 쇼핑몰"): 3, ("쇼핑","충동구매"): 2,
    ("문화/여가","영화/공연"): 2, ("문화/여가","게임/콘텐츠"): 3, ("문화/여가","취미/오락"): 2,
    ("문화/여가","여행"): 1,
    ("기타","기타"): 2,
}

total_txns = 80
keys = list(weights.keys())
probs = np.array([weights[k] for k in keys], dtype=float)
probs = probs / probs.sum()
chosen = rng.choice(len(keys), size=total_txns, p=probs)

records = []
times = sample_txn_times(total_txns, late_night_ratio=0.28, weekend_ratio=0.5)
for i, idx in enumerate(chosen):
    big, small = keys[idx]
    merchant = rng.choice(merchants_by_sub[(big, small)])
    amt = sample_amount(small)
    status = "성공" if rng.random() > 0.06 else "취소"
    created = times[i] - timedelta(minutes=int(rng.integers(0, 120)))
    updated = times[i] + timedelta(minutes=int(rng.integers(0, 120)))
    records.append({
        "사용자 아이디": user_id,
        "대분류 카테고리": big,
        "소분류 카테고리": small,
        "거래시간": times[i],
        "금액": amt,
        "거래처": merchant,
        "상태": status,
        "생성일": created,
        "수정일": updated,
    })

tx = pd.DataFrame.from_records(records)
tx = tx.sort_values("거래시간").reset_index(drop=True)

# ==== 1) Data modeling & normalization (filter, cast, basic flags) ====
tx["is_success"] = (tx["상태"] == "성공")
tx_succ = tx[tx["is_success"]].copy()

tx_succ["is_weekend"] = tx_succ["거래시간"].dt.weekday >= 5
tx_succ["hour"] = tx_succ["거래시간"].dt.hour
tx_succ["is_late_night"] = ((tx_succ["hour"] >= 22) | (tx_succ["hour"] < 6))

# ==== 2) Feature engineering for the last 30 days (single user) ====
# Total spend
total_spend_30 = tx_succ["금액"].sum()

# Per subcategory aggregates
grp = tx_succ.groupby(["소분류 카테고리"], as_index=False).agg(
    spend_30d=("금액","sum"),
    freq_30d=("금액","size"),
    avg_ticket=("금액","mean"),
    max_ticket=("금액","max"),
    late_night_share=("is_late_night","mean"),
    weekend_share=("is_weekend","mean")
)
grp["share_30d"] = grp["spend_30d"] / (total_spend_30 + 1e-9)

# Recency: last 7 days vs 30 days
cut_7d = now - timedelta(days=7)
grp7 = tx_succ[tx_succ["거래시간"] >= cut_7d].groupby("소분류 카테고리", as_index=False)["금액"].sum().rename(columns={"금액":"spend_7d"})
grp = grp.merge(grp7, on="소분류 카테고리", how="left").fillna({"spend_7d":0})
grp["recency7d_share"] = grp["spend_7d"] / (grp["spend_30d"] + 1e-9)

# Trend: last 14d vs previous 14d
cut_14 = now - timedelta(days=14)
last14 = tx_succ[tx_succ["거래시간"] >= cut_14].groupby("소분류 카테고리", as_index=False)["금액"].sum().rename(columns={"금액":"spend_last14"})
prev14 = tx_succ[(tx_succ["거래시간"] < cut_14) & (tx_succ["거래시간"] >= now - timedelta(days=28))]\
    .groupby("소분류 카테고리", as_index=False)["금액"].sum().rename(columns={"금액":"spend_prev14"})

grp = grp.merge(last14, on="소분류 카테고리", how="left").merge(prev14, on="소분류 카테고리", how="left").fillna(0)
grp["trend14d_pct"] = np.where(
    grp["spend_prev14"] > 0,
    (grp["spend_last14"] - grp["spend_prev14"]) / grp["spend_prev14"],
    np.nan
)

# Global behavioral features
late_night_share_all = tx_succ["is_late_night"].mean()
weekend_share_all = tx_succ["is_weekend"].mean()
avg_ticket_all = tx_succ["금액"].mean()

# Compile a per-user feature vector (selected features for clustering demo)
def share_of(sub):
    row = grp.loc[grp["소분류 카테고리"]==sub]
    return float(row["share_30d"]) if not row.empty else 0.0

def share_of_group(subs):
    return float(grp.loc[grp["소분류 카테고리"].isin(subs), "share_30d"].sum())

feat_user = {
    "user_id": user_id,
    # key category shares
    "share_coffee": share_of("커피"),
    "share_delivery": share_of("배달음식"),
    "share_dineout": share_of("외식"),
    "share_snack": share_of("간식"),
    "share_ott": share_of("OTT/구독서비스"),
    "share_game_content": share_of_group(["게임/콘텐츠","영화/공연","취미/오락"]),
    "share_shopping": share_of_group(["의류/패션","뷰티/미용","온라인 쇼핑몰","충동구매"]),
    # global behavior
    "late_night_share_all": late_night_share_all,
    "weekend_share_all": weekend_share_all,
    "avg_ticket_all": avg_ticket_all,
}

user_feat_df = pd.DataFrame([feat_user])

# ==== 3) Simple clustering demo with synthetic peers ====
# Build a small peer set around different archetypes
def make_peer(uid, base, noise=0.05, ln=None, wk=None, avg=None):
    vec = base.copy()
    # add small gaussian noise and re-normalize cat shares to sum to <=1
    keys = ["share_coffee","share_delivery","share_dineout","share_snack",
            "share_ott","share_game_content","share_shopping"]
    for k in keys:
        vec[k] = max(0.0, base[k] + rng.normal(0, noise))
    total_share = sum(vec[k] for k in keys)
    if total_share > 0:
        for k in keys:
            vec[k] = vec[k]/total_share * min(total_share, 0.95)  # cap a bit
    vec["late_night_share_all"] = ln if ln is not None else np.clip(base.get("late_night_share_all",0.2)+rng.normal(0,0.08), 0, 1)
    vec["weekend_share_all"] = wk if wk is not None else np.clip(base.get("weekend_share_all",0.5)+rng.normal(0,0.08), 0, 1)
    vec["avg_ticket_all"] = avg if avg is not None else max(5000, base.get("avg_ticket_all",20000)+rng.normal(0,5000))
    vec["user_id"] = uid
    return vec

# Archetypes
A_coffee_delivery = {"share_coffee":0.2,"share_delivery":0.28,"share_dineout":0.18,"share_snack":0.12,
                     "share_ott":0.03,"share_game_content":0.07,"share_shopping":0.12,
                     "late_night_share_all":0.25,"weekend_share_all":0.55,"avg_ticket_all":23000}
B_dineout_alcohol = {"share_coffee":0.08,"share_delivery":0.12,"share_dineout":0.42,"share_snack":0.08,
                     "share_ott":0.03,"share_game_content":0.07,"share_shopping":0.20,
                     "late_night_share_all":0.18,"weekend_share_all":0.6,"avg_ticket_all":30000}
C_shopping_heavy = {"share_coffee":0.06,"share_delivery":0.08,"share_dineout":0.12,"share_snack":0.06,
                    "share_ott":0.04,"share_game_content":0.06,"share_shopping":0.52,
                    "late_night_share_all":0.15,"weekend_share_all":0.48,"avg_ticket_all":45000}
D_subscriptions_gamer = {"share_coffee":0.10,"share_delivery":0.10,"share_dineout":0.10,"share_snack":0.06,
                         "share_ott":0.18,"share_game_content":0.32,"share_shopping":0.14,
                         "late_night_share_all":0.35,"weekend_share_all":0.5,"avg_ticket_all":20000}

peers = []
for i in range(2):
    peers.append(make_peer(f"peer_A_{i+1}", A_coffee_delivery))
for i in range(2):
    peers.append(make_peer(f"peer_B_{i+1}", B_dineout_alcohol))
for i in range(2):
    peers.append(make_peer(f"peer_C_{i+1}", C_shopping_heavy))
for i in range(2):
    peers.append(make_peer(f"peer_D_{i+1}", D_subscriptions_gamer))

peer_df = pd.DataFrame(peers)
all_feat = pd.concat([user_feat_df, peer_df], ignore_index=True)

feature_cols = ["share_coffee","share_delivery","share_dineout","share_snack",
                "share_ott","share_game_content","share_shopping",
                "late_night_share_all","weekend_share_all","avg_ticket_all"]

scaler = StandardScaler()
X = scaler.fit_transform(all_feat[feature_cols])
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
all_feat["cluster"] = labels

# Name clusters by top share components
centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_cols)
centers_unscaled = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)

def name_cluster(row):
    # rank by category shares (ignore behavior cols when ranking)
    cat_cols = ["share_coffee","share_delivery","share_dineout","share_snack",
                "share_ott","share_game_content","share_shopping"]
    top3 = row[cat_cols].sort_values(ascending=False).index[:3]
    name_map = {
        "share_coffee":"커피",
        "share_delivery":"배달",
        "share_dineout":"외식",
        "share_snack":"간식",
        "share_ott":"구독",
        "share_game_content":"게임·콘텐츠",
        "share_shopping":"쇼핑",
    }
    return " · ".join([name_map[t] for t in top3])

cluster_labels = centers_unscaled.apply(name_cluster, axis=1).to_dict()
all_feat["cluster_name"] = all_feat["cluster"].map(cluster_labels)

# Build compact outputs to display
tx_show = tx.head(20)  # only show first 20 rows for brevity
summary_by_subcat = grp.sort_values("spend_30d", ascending=False).reset_index(drop=True)
user_row = all_feat[all_feat["user_id"] == user_id].reset_index(drop=True)
clusters_overview = all_feat[["user_id","cluster","cluster_name"]].sort_values("cluster").reset_index(drop=True)

# Provide DataFrames to the user
print("샘플 거래내역(일부)", tx_show)
print("사용자별 서브카테고리 요약 (30일)", summary_by_subcat)
print("클러스터 배정 결과", clusters_overview)
print("내 사용자 피처 벡터 (요약)", user_row[["user_id"]+feature_cols+["cluster_name"]])


