from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np

TFIDF_VEC = None
TFIDF_CLF = None
KOBERT_TOK = None
KOBERT_MODEL = None
KOBERT_CLF = None

ML_CONFIRM = float(os.getenv("ML_CONFIRM", "0.75"))

__INITIALIZED = False
__LABEL_NAMES: Optional[List[str]] = None

ARTIFACT_DIR = Path(os.getenv("MODEL_DIR", "./artifacts")).resolve()
VEC_PATH  = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
CLF_PATH  = ARTIFACT_DIR / "tfidf_model.joblib"
LAB_PATH  = ARTIFACT_DIR / "labels.json"

# KoBERT artifacts
TOKENIZER_DIR = ARTIFACT_DIR / "kobert_tokenizer"
BASEMODEL_DIR = ARTIFACT_DIR / "kobert_model"
KOBERT_CLF_PATH = ARTIFACT_DIR / "kobert_clf.joblib"

# (lazy) torch/transformers 임포트: CPU환경에서도 안전
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
    categorization.py에서 여러 번 호출해도 안전.  :contentReference[oaicite:11]{index=11}
    """
    global __INITIALIZED, TFIDF_VEC, TFIDF_CLF, __LABEL_NAMES
    global KOBERT_TOK, KOBERT_MODEL, KOBERT_CLF

    if __INITIALIZED:
        return

    # TF-IDF
    if VEC_PATH.exists() and CLF_PATH.exists():
        try:
            TFIDF_VEC = joblib.load(VEC_PATH)
            TFIDF_CLF = joblib.load(CLF_PATH)
        except Exception as e:
            print(f"[WARN] Failed to load TF-IDF artifacts: {e}")

    # 공용 라벨
    if LAB_PATH.exists():
        try:
            __LABEL_NAMES = json.loads(LAB_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to load labels.json: {e}")
            __LABEL_NAMES = None

    # KoBERT (선택)
    try:
        if TOKENIZER_DIR.exists() and BASEMODEL_DIR.exists() and KOBERT_CLF_PATH.exists():
            _lazy_import_torch()
            KOBERT_TOK = _AutoTokenizer.from_pretrained(TOKENIZER_DIR.as_posix())
            KOBERT_MODEL = _AutoModel.from_pretrained(BASEMODEL_DIR.as_posix()).to(_DEVICE)
            KOBERT_MODEL.eval()
            KOBERT_CLF = joblib.load(KOBERT_CLF_PATH)
    except Exception as e:
        print(f"[WARN] Failed to load KoBERT artifacts: {e}")
        KOBERT_TOK = None
        KOBERT_MODEL = None
        KOBERT_CLF = None

    __INITIALIZED = True

def get_label_names() -> Optional[List[str]]:
    return __LABEL_NAMES

def kobert_embed(texts: List[str], max_len: int = 64) -> Optional[np.ndarray]:
    """
    KoBERT CLS 임베딩 추출 → numpy (N, hidden)
    categorization.MLRule에서 사용.  :contentReference[oaicite:12]{index=12}
    """
    if KOBERT_TOK is None or KOBERT_MODEL is None:
        return None
    _lazy_import_torch()
    with _torch.no_grad():
        all_embs = []
        # 간단히 배치=64
        bs = 64
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            toks = KOBERT_TOK(
                batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            ).to(_DEVICE)
            out = KOBERT_MODEL(**toks)
            cls = out.last_hidden_state[:, 0, :]   # CLS
            all_embs.append(cls.detach().cpu().numpy())
        return np.vstack(all_embs)
