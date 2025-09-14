from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from database import make_engine_from_env, TransactionRepository
# from utils import build_user_features
from features import build_user_features
from visualization import check_n_with_elbow, visualization
from visualization import cluster_scatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=200, help="최근 N일 범위")
    parser.add_argument("--best-k", type=int, default=None, help="실루엣 플롯에 사용할 K(없으면 자동 보정)")
    parser.add_argument("--user-id", type=int, default=None, help="특정 사용자만 분석(옵션)")

    parser.add_argument("--embed", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()

    engine = make_engine_from_env()
    repo = TransactionRepository(engine)
    tx = repo.fetch_last_days(days=args.days, user_id=args.user_id)

    if tx.empty:
        print(f"[WARN] 최근 {args.days}일 데이터가 없습니다.")
        return

    user_feat = build_user_features(tx)
    if user_feat.shape[0] < 2:
        print("[WARN] 사용자 수가 2명 미만이라 클러스터링을 진행할 수 없습니다.")
        return

    # 표준화: 각 피처 스케일 정규화
    scaler = StandardScaler()
    X = scaler.fit_transform(user_feat.values)

    # 1) 적정 K 탐색
    check_n_with_elbow(X)

    # 2) 실루엣 플롯
    visualization(X, best_k=args.best_k)

    # 1) K 선택(직접 지정 없으면 샘플 수 기반으로 합리적 기본값)
    if args.best_k is None:
        n_samples = X.shape[0]
        best_k = max(2, min(6, n_samples - 1))
    else:
        best_k = args.best_k

    # 2) KMeans 라벨링
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # 3) 결과 저장(유저-클러스터 매핑)
    os.makedirs("outputs", exist_ok=True)
    assign = pd.DataFrame({"user_id": user_feat.index, "cluster": labels})
    assign_path = os.path.join("outputs", "cluster_assignments.csv")
    assign.to_csv(assign_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] cluster assignments saved → {assign_path}")

    # 4) 산점도(사용자별 그룹 확인)
    #    - PCA가 빠르고 안정적, 자세한 라벨 표시가 필요하면 annotate=True
    plot_path = os.path.join("outputs", f"cluster_scatter_k{best_k}.png")
    cluster_scatter(
        X, labels,
        user_ids=list(user_feat.index),
        method="pca",  # or "tsne"
        annotate=False,  # 사용자 ID를 점 위에 표기하려면 True
        savepath=plot_path
    )
    print(f"[INFO] scatter saved → {plot_path}")

if __name__ == "__main__":
    main()
