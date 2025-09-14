from sklearn.preprocessing import StandardScaler

from utils import *
from visualization import *


def main() :
    # 표준화
    scaler = StandardScaler()
    X = scaler.fit_transform(user_feat)

    # 1) 적정 K 찾기 (엘보 + 실루엣)
    check_n_with_elbow(X)

    # 2) 선택한 K로 실루엣 플롯
    best_k = 4  # check_n_with_elbow 그래프 보고 수정
    visualization(X, best_k=best_k)


if __name__ == "__main__":
    main()
