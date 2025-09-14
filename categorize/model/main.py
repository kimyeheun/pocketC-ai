# pocketc/model/main.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional, List, Tuple

import joblib
import numpy as np

# ===== 공개 심볼 =====
TFIDF_VEC = None          # sklearn Vectorizer
TFIDF_CLF = None          # sklearn Classifier
KOBERT_TOK = None         # KoBERT tokenizer (optional)
KOBERT_MODEL = None       # KoBERT base model (optional)
KOBERT_CLF = None         # KoBERT classifier head (optional)

ML_CONFIRM = float(os.getenv("ML_CONFIRM", "0.75"))

# 내부 상태
__INITIALIZED = False
__LABEL_NAMES: Optional[List[str]] = None

# 아티팩트 경로 규약 (학습 스크립트와 동일)
ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", "./artifacts")).resolve()
VEC_PATH      = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
CLF_PATH      = ARTIFACT_DIR / "tfidf_model.joblib"
LAB_PATH      = ARTIFACT_DIR / "labels.json"

# KoBERT artifacts (옵션)
TOKENIZER_DIR   = ARTIFACT_DIR / "kobert_tokenizer"
BASEMODEL_DIR   = ARTIFACT_DIR / "kobert_model"
KOBERT_CLF_PATH = ARTIFACT_DIR / "kobert_clf.joblib"

# (lazy) torch/transformers import (CPU 환경 고려)
_torch = None
_AutoTokenizer = None
_AutoModel = None
_DEVICE = None

def _lazy_import_torch():
    global _torch, _AutoTokenizer, _AutoModel, _DEVICE
    if _torch is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModel
    _torch = torch
    _AutoTokenizer = AutoTokenizer
    _AutoModel = AutoModel
    _DEVICE = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

def init_ml_once() -> None:
    """
    서비스 전역에서 여러 번 호출해도 안전한 레이지 로더.
    - TF-IDF 벡터라이저/분류기
    - labels.json
    - (있으면) KoBERT 토크나이저/베이스모델/분류기
    """
    global __INITIALIZED, TFIDF_VEC, TFIDF_CLF, __LABEL_NAMES
    global KOBERT_TOK, KOBERT_MODEL, KOBERT_CLF

    if __INITIALIZED:
        return

    # TF-IDF artifacts
    if VEC_PATH.exists() and CLF_PATH.exists():
        try:
            TFIDF_VEC = joblib.load(VEC_PATH)
            TFIDF_CLF = joblib.load(CLF_PATH)
        except Exception as e:
            print(f"[WARN] Failed to load TF-IDF artifacts: {e}")

    # label names
    if LAB_PATH.exists():
        try:
            __LABEL_NAMES = json.loads(LAB_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to load labels.json: {e}")
            __LABEL_NAMES = None

    # KoBERT (optional)
    try:
        if TOKENIZER_DIR.exists() and BASEMODEL_DIR.exists() and KOBERT_CLF_PATH.exists():
            _lazy_import_torch()
            KOBERT_TOK   = _AutoTokenizer.from_pretrained(TOKENIZER_DIR.as_posix())
            KOBERT_MODEL = _AutoModel.from_pretrained(BASEMODEL_DIR.as_posix()).to(_DEVICE)
            KOBERT_MODEL.eval()
            KOBERT_CLF   = joblib.load(KOBERT_CLF_PATH)
    except Exception as e:
        print(f"[WARN] Failed to load KoBERT artifacts: {e}")
        KOBERT_TOK = None
        KOBERT_MODEL = None
        KOBERT_CLF = None

    __INITIALIZED = True

def get_label_names() -> Optional[List[str]]:
    """학습 시 저장된 labels.json이 있으면 리스트 반환, 없으면 None."""
    return __LABEL_NAMES

def kobert_embed(texts: List[str], max_len: int = 64) -> Optional[np.ndarray]:
    """
    KoBERT CLS 임베딩 → numpy (N, hidden)
    KOBERT 토크나이저/모델이 로드된 경우에만 동작, 아니면 None 반환.
    """
    if KOBERT_TOK is None or KOBERT_MODEL is None:
        return None
    _lazy_import_torch()
    with _torch.no_grad():
        all_embs = []
        bs = 64
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            toks = KOBERT_TOK(
                batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            ).to(_DEVICE)
            out = KOBERT_MODEL(**toks)
            cls = out.last_hidden_state[:, 0, :]
            all_embs.append(cls.detach().cpu().numpy())
        return np.vstack(all_embs)

def predict_candidates(merchant: str, label_inverse_map: Optional[List[str]]) -> List[Tuple[str, str, float]]:
    """
    규칙 체인에서 호출되는 통합 예측 어댑터:
    - TF-IDF clf → 최고 확률이 ML_CONFIRM 이상이면 후보 추가
    - KoBERT clf → 동일
    반환: [(sub_name, source, confidence), ...]
    """
    init_ml_once()
    cands: List[Tuple[str, str, float]] = []

    # TF-IDF
    if TFIDF_VEC is not None and TFIDF_CLF is not None:
        try:
            X = TFIDF_VEC.transform([merchant])
            if hasattr(TFIDF_CLF, "predict_proba"):
                p = TFIDF_CLF.predict_proba(X)[0]
            else:
                d = TFIDF_CLF.decision_function(X)[0]
                d = d - d.max()
                p = np.exp(d) / np.exp(d).sum()
            idx = int(np.argmax(p)); conf = float(p[idx])
            if conf >= ML_CONFIRM:
                name = (label_inverse_map[idx] if label_inverse_map and 0 <= idx < len(label_inverse_map) else "기타")
                cands.append((name, "model:tfidf", conf))
        except Exception:
            pass

    # KoBERT
    if KOBERT_TOK is not None and KOBERT_MODEL is not None and KOBERT_CLF is not None:
        try:
            emb = kobert_embed([merchant])
            if emb is not None:
                p = KOBERT_CLF.predict_proba(emb)[0]
                idx = int(np.argmax(p)); conf = float(p[idx])
                if conf >= ML_CONFIRM:
                    name = (label_inverse_map[idx] if label_inverse_map and 0 <= idx < len(label_inverse_map) else "기타")
                    cands.append((name, "model:kobert", conf))
        except Exception:
            pass

    return cands
