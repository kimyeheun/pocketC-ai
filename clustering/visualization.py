import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# 시스템에 폰트 없을 수 있으므로 필요 시만 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

def _dynamic_k_range(n_samples, k_min=2, k_max=10):
    print(n_samples)
    upper = min(k_max, max(k_min, n_samples - 1))
    lower = min(k_min, upper)
    return range(lower, upper + 1)

def check_n_with_elbow(X, k_max=10):
    n_samples = X.shape[0]
    K_range = _dynamic_k_range(n_samples, k_min=2, k_max=k_max)

    if len(K_range) == 0:
        print("[check_n_with_elbow] 샘플 수가 너무 적어(K<2 불가) K 탐색을 생략합니다.")
        return

    sse = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        sse.append(km.inertia_)

    plt.figure(figsize=(6,4))
    plt.plot(list(K_range), sse, marker='o')
    plt.xlabel("클러스터 개수 (K)")
    plt.ylabel("SSE (Within-Cluster Sum of Squares)")
    plt.title("엘보 방법 (Elbow Method)")
    plt.show()

    sil_scores = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        n_labels = len(np.unique(labels))
        if n_labels < 2 or n_labels > n_samples - 1:
            sil_scores.append(np.nan)
            continue
        sil = silhouette_score(X, labels)
        sil_scores.append(sil)

    plt.figure(figsize=(6,4))
    plt.plot(list(K_range), sil_scores, marker='o')
    plt.xlabel("클러스터 개수 (K)")
    plt.ylabel("평균 실루엣 점수")
    plt.title("실루엣 분석 (Silhouette Score)")
    plt.show()

def visualization(X, best_k=None):
    n_samples = X.shape[0]
    K_range = _dynamic_k_range(n_samples, k_min=2, k_max=10)

    if len(K_range) == 0:
        print("[visualization] 샘플 수가 너무 적어 실루엣 플롯을 그릴 수 없습니다.")
        return

    if best_k is None or best_k < min(K_range) or best_k > max(K_range):
        best_k = min(4, max(K_range))

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > n_samples - 1:
        print(f"[visualization] K={best_k}에서 실루엣 제약 위반 → 다른 K로 시도하세요. (n_samples={n_samples})")
        return

    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score (K={best_k}): {silhouette_avg:.3f}")

    sample_silhouette_values = silhouette_samples(X, labels)

    fig, ax = plt.subplots(figsize=(7,5))
    y_lower = 10
    for i in range(best_k):
        ith = sample_silhouette_values[labels == i]
        ith.sort()
        size_i = ith.shape[0]
        y_upper = y_lower + size_i
        color = cm.nipy_spectral(float(i) / best_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlabel("실루엣 계수 값")
    ax.set_ylabel("클러스터 레이블")
    ax.set_title(f"Silhouette Plot (K={best_k})")
    plt.show()



def cluster_scatter(X, labels, user_ids, method: str = "pca",
                    annotate: bool = False, figsize=(7,5), savepath: str | None = None):
    """
    X: (n_samples, n_features) - 이미 스케일링된 행렬
    labels: (n_samples,) - KMeans 등으로 얻은 군집 라벨
    user_ids: (n_samples,) - 각 점에 표시할 식별자(숫자/문자열)
    method: "pca" | "tsne"  (기본 pca가 빠르고 안정적)
    """
    n_clusters = int(np.max(labels)) + 1

    if method.lower() == "tsne":
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
        title = f"t-SNE Scatter (K={n_clusters})"
    else:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=42).fit_transform(X)
        title = f"PCA Scatter (K={n_clusters})"

    fig, ax = plt.subplots(figsize=figsize)
    for k in range(n_clusters):
        mask = (labels == k)
        color = cm.nipy_spectral(float(k) / max(1, n_clusters))
        ax.scatter(Z[mask, 0], Z[mask, 1], s=40, alpha=0.8, label=f"cluster {k}", color=color, edgecolors="none")

        # 필요 시 클러스터 중심도 표시
        if np.any(mask):
            cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
            ax.scatter([cx], [cy], s=220, marker="X", color=color, edgecolors="black", linewidths=0.5, zorder=3)

    # 사용자 ID 라벨 달기(많을 때는 시야가 지저분해지므로 옵션)
    if annotate:
        for (x, y), uid in zip(Z, user_ids):
            ax.text(x, y, str(uid), fontsize=8, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)

    if savepath:
        plt.savefig(savepath, bbox_inches="tight", dpi=150)
    plt.show()
