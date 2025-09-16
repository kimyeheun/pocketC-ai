import joblib
import json
from pathlib import Path

from common.engine import make_engine_from_env
from database import TransactionRepository
from features import build_user_features

cluster_path = Path("cluster")

def predict_cluster_for_user(user_id: int, days: int = 30) -> int | None:
    # 1) 아티팩트 로드
    scaler = joblib.load(cluster_path / "scaler.joblib")
    kmeans = joblib.load(cluster_path / "kmeans.joblib")
    feature_cols = json.loads((cluster_path / "feature_cols.json").read_text(encoding="utf-8"))

    # 2) 새 유저 거래 로드
    engine = make_engine_from_env()
    repo = TransactionRepository(engine)
    tx = repo.fetch_last_days(days=days, user_id=user_id)  # ← 특정 사용자만 로드 :contentReference[oaicite:7]{index=7}
    if tx.empty:
        return None

    # 3) 동일 로직으로 피처 생성 (카테고리 유니버스는 feature_cols에 이미 내재)
    user_feat = build_user_features(tx)  # 새 유저 단 1행이어도 OK :contentReference[oaicite:8]{index=8}

    # 4) 학습 컬럼에 맞춰 정렬/보정
    X = user_feat.reindex(columns=feature_cols, fill_value=0.0).values

    # 5) 스케일링 + 예측
    Xs = scaler.transform(X)   # ← 학습 스케일러로 변환  :contentReference[oaicite:9]{index=9}
    cluster = int(kmeans.predict(Xs)[0])  # ← KMeans로 클러스터 할당  :contentReference[oaicite:10]{index=10}
    return cluster

if __name__ == "__main__":
    predict = predict_cluster_for_user(user_id=1, days=200)
    print(predict)
