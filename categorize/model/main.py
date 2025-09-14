import os
from typing import List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer # noqa
from sklearn.linear_model import LogisticRegression # noqa
from transformers import AutoTokenizer, AutoModel  # noqa
import torch

TFIDF_VEC = None
TFIDF_CLF = None
KOBERT_TOK = None
KOBERT_MODEL = None
KOBERT_CLF = None


MODEL_DIR = os.getenv("MODEL_DIR", "./models")
USE_TFIDF = os.getenv("USE_TFIDF", "0") == "1"
USE_KOBERT = os.getenv("USE_KOBERT", "0") == "1"
ML_CONFIRM = float(os.getenv("ML_CONFIRM", "0.75"))


_ML_INIT_DONE = False


def init_ml_once(type: bool):
    global _ML_INIT_DONE, TFIDF_VEC, TFIDF_CLF, KOBERT_TOK, KOBERT_MODEL, KOBERT_CLF
    if _ML_INIT_DONE:
        return
    if type:
        try:
            TFIDF_VEC = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))
            TFIDF_CLF = joblib.load(os.path.join(MODEL_DIR, 'tfidf_logreg.joblib'))
            print('[INFO] TF-IDF model loaded')
        except Exception as e:
            print('[WARN] TF-IDF load failed:', e)
    else:
        try:
            KOBERT_TOK = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
            KOBERT_MODEL = AutoModel.from_pretrained('skt/kobert-base-v1')
            KOBERT_CLF = joblib.load(os.path.join(MODEL_DIR, 'kobert_logreg.joblib'))
            KOBERT_MODEL.eval()
            print('[INFO] KoBERT classifier loaded')
        except Exception as e:
            print('[WARN] KoBERT load failed:', e)
    _ML_INIT_DONE = True


def kobert_embed(texts: List[str]):
    inputs = KOBERT_TOK(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        out = KOBERT_MODEL(**inputs).last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean = summed / counts
    return mean.numpy()