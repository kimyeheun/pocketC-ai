# visualization.py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

plt.rcParams['font.family'] = 'Malgun Gothic'   # 환경에 없는 폰트면 주석 처리하세요
plt.rcParams['axes.unicode_minus'] = False

def _dynamic_k_range(n_samples, k_min=2, k_max=10):
    """
    n_samples 기준으로 가능한 K 범위를 동적으로 산출
    실루엣 계산 제약: 2 <= K <= n_samples-1
    """
    upper = min(k_max, max(k_min, n_samples - 1))
    lower = min(k_min, upper)  # n_samples<3이면 lower==upper
    return range(lower, upper + 1)

def check_n_with_elbow(X, k_max=10):
    n_samples = X.shape[0]
    K_range = _dynamic_k_range(n_samples, k_min=2, k_max=k_max)

    if len(K_range) == 0:
        print("[check_n_with_elbow] 샘플 수가 너무 적어(K<2 불가) K 탐색을 생략합니다.")
        return

    # 1) 엘보
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

    # 2) 실루엣 (가능한 경우에만)
    sil_scores = []
    for k in K_range:
        # 실루엣 제약: n_unique_labels ∈ [2, n_samples-1]
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

    # best_k가 없거나 범위를 벗어나면, K_range 중간값(혹은 최댓값)으로 보정
    if best_k is None or best_k < min(K_range) or best_k > max(K_range):
        best_k = min(4, max(K_range))  # 선호 기본값 4, 불가하면 가능한 최댓값으로

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels > n_samples - 1:
        print(f"[visualization] K={best_k}에서 실루엣 제약 위반 → 다른 K로 시도하세요. (n_samples={n_samples})")
        return

    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score (K={best_k}): {silhouette_avg:.3f}")

    sample_silhouette_values = silhouette_samples(X, labels)

    import matplotlib.cm as cm
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
